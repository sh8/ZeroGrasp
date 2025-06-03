import copy

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import ocnn
from ocnn.octree import Points, Octree
from torch_scatter import scatter_max
from ofe import OctreeFeatureExtractor

from zerograsp.utils.math import unnormalize_pts, project_to_image_plane
from zerograsp.nets.utils import (
    get_xyz_from_octree,
    octree_align,
    merge_octrees,
    octree_unnormalize_pts,
)
from zerograsp.nets.mae_encoder import MAEEncoder
from zerograsp.nets.ocnn_blocks import (
    OctreeConvBnElu,
    OctreeDeconvBnElu,
    OctreeResBlocks,
    Conv1x1BnElu,
)
from zerograsp.nets.conv3d_encoder import Conv3DEncoder


class ZeroGrasp(pl.LightningModule):
    def __init__(self, config) -> None:
        super(ZeroGrasp, self).__init__()
        self.config = config
        self.backbone_model = config.backbone_model
        self.use_align_corners = config.use_align_corners
        self.mode_sample = config.mode_sample

        self.dim_model = config.dim_model
        self.dim_mae = config.dim_mae
        self.max_lod = config.max_lod
        self.min_lod = config.min_lod
        self.grid_res = 1 << self.min_lod
        self.grid_size = config.grid_size
        self.pe_type = config.pe_type
        self.attn_type = config.attn_type
        self.mae_type = config.mae_type
        self.update_octree = config.update_octree
        self.input_feature_type = config.input_feature_type
        self.use_mult_obj_enc = config.use_mult_obj_enc
        self.use_sing_obj_ref = config.use_sing_obj_ref
        self.use_pool_and_concat = config.use_pool_and_concat
        self.use_per_obj_latent = config.use_per_obj_latent
        self.use_cvae = config.use_cvae
        self.use_full_layer = config.use_full_layer
        self.concat_obj_latent = config.concat_obj_latent
        self.ofe_use_conv3d = config.ofe_use_conv3d
        self.use_only_visible = config.use_only_visible

        self.channel_in = 0
        if "F" in self.input_feature_type:
            if self.backbone_model is not None:
                self.channel_in += self.dim_model
            else:
                self.channel_in += 3
        if "L" in self.input_feature_type:
            self.channel_in += 3
        if "P" in self.input_feature_type:
            self.channel_in += 3
        # if 'V' in self.input_feature_type:
        #     self.channel_in += 3

        self.resblk_num = 2
        self.channels = [512, 512, 256, 256, 192, 192, 96, 96, 48, 48]
        if config.predict_grasp:
            self.channels_out = [2, 2, 2, 2, 2, 2, 2, 2, 2, 16]
            self.predict_hidden_dims = [64, 64, 64, 64, 64, 64, 64, 64, 64, 256]
        else:
            self.channels_out = [2, 2, 2, 2, 2, 2, 2, 2, 2, 6]
            self.predict_hidden_dims = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]

        if self.backbone_model == "resnext":
            self.backbone = smp.Unet(
                encoder_name="resnext50_32x4d",
                encoder_weights="imagenet",
                in_channels=3,
                classes=config.dim_model,
            )
            for param in self.backbone.encoder.parameters():
                param.requires_grad = False
            for param in self.backbone.decoder.parameters():
                param.requires_grad = False
        else:
            self.backbone = None

        # if self.use_mult_obj_enc:
        self.mae_encoder = MAEEncoder(config)

        if self.use_sing_obj_ref:
            self.single_obj_refine = MAEEncoder(config)

        if self.use_cvae:
            self.channel_latent = config.latent_dim
        else:
            self.channel_latent = 0

        # 3D mask encoding
        aux_channels = [0] * (self.max_lod + 1)
        if "M" in self.input_feature_type:
            self.ofe_multiple = 3
            self.ofe_grid_size = 10.0
            self.ofe_quad_multiple = 1 << (self.ofe_multiple)
            self.ofe_cubic_multiple = 1 << (3 * self.ofe_multiple)
            offset = (
                th.arange(self.ofe_quad_multiple) / self.ofe_quad_multiple
                + 1 / (2 * self.ofe_quad_multiple)
                - 0.5
            )
            ofe_offset = self.grid_size * th.stack(
                th.meshgrid(offset, offset, offset, indexing="ij"), dim=-1
            ).reshape(1, -1, 3)
            self.register_buffer("ofe_offset", ofe_offset)
            self.ofe = OctreeFeatureExtractor(config.img_height, config.img_width)
            if self.ofe_use_conv3d:
                self.ofe_out_channel = 64
                self.mask_head = Conv3DEncoder(
                    2, self.ofe_out_channel, f_maps=[8, 16, 32, 64]
                )
            else:
                self.ofe_out_channel = self.ofe_cubic_multiple // 4
                self.mask_head = nn.Sequential(
                    ocnn.modules.FcBnRelu(
                        self.ofe_cubic_multiple * 2, self.ofe_cubic_multiple
                    ),
                    ocnn.modules.FcBnRelu(
                        self.ofe_cubic_multiple, self.ofe_cubic_multiple
                    ),
                    nn.Linear(self.ofe_cubic_multiple, self.ofe_out_channel),
                )
            aux_channels[self.min_lod] += self.ofe_out_channel

        # encoder
        self.conv1 = OctreeConvBnElu(
            self.channel_in, self.channels[self.max_lod], nempty=False
        )
        self.conv2 = OctreeConvBnElu(
            self.channels[self.min_lod] + self.channel_latent,
            self.channels[self.min_lod],
            nempty=False,
        )
        # self.conv2 = OctreeResBlocks(
        #     self.channels[self.min_lod] + self.channel_latent, self.channels[self.min_lod], self.resblk_num, nempty=False)
        self.encoder_blks = nn.ModuleList(
            [
                OctreeResBlocks(
                    self.channels[d] + aux_channels[d],
                    self.channels[d],
                    self.resblk_num,
                    nempty=False,
                )
                for d in range(self.max_lod, self.min_lod - 1, -1)
            ]
        )
        self.downsample = nn.ModuleList(
            [
                OctreeConvBnElu(
                    self.channels[d],
                    self.channels[d - 1],
                    kernel_size=[2],
                    stride=2,
                    nempty=False,
                )
                for d in range(self.max_lod, self.min_lod, -1)
            ]
        )

        # decoder
        self.upsample = nn.ModuleList(
            [
                OctreeDeconvBnElu(
                    self.channels[d - 1],
                    self.channels[d],
                    kernel_size=[2],
                    stride=2,
                    nempty=False,
                )
                for d in range(self.min_lod + 1, self.max_lod + 1)
            ]
        )
        self.decoder_blks = nn.ModuleList(
            [
                OctreeResBlocks(
                    self.channels[d], self.channels[d], self.resblk_num, nempty=False
                )
                for d in range(self.min_lod, self.max_lod + 1)
            ]
        )

        # header
        self.predict = nn.ModuleList(
            [
                self._make_predict_module(
                    self.channels[d], self.channels_out[d], self.predict_hidden_dims[d]
                )
                for d in range(self.min_lod, self.max_lod + 1)
            ]
        )

        # prior, posterior
        if self.use_cvae:
            self.posterior_conv1 = OctreeConvBnElu(
                14, self.channels[self.max_lod], nempty=False
            )  # Normals (3 channels) + Local coordinates (3 channels) + Displacement (1 channel)
            self.posterior_encoder_blks = nn.ModuleList(
                [
                    OctreeResBlocks(
                        self.channels[d],
                        self.channels[d],
                        self.resblk_num,
                        nempty=False,
                    )
                    for d in range(self.max_lod, self.min_lod - 1, -1)
                ]
            )
            self.posterior_downsample = nn.ModuleList(
                [
                    OctreeConvBnElu(
                        self.channels[d],
                        self.channels[d - 1],
                        kernel_size=[2],
                        stride=2,
                        nempty=False,
                    )
                    for d in range(self.max_lod, self.min_lod, -1)
                ]
            )
            if self.concat_obj_latent:
                posterior_channel_in = self.channels[self.min_lod] * 2
            else:
                posterior_channel_in = self.channels[self.min_lod]
            if self.use_per_obj_latent:
                self.prior_head = nn.Sequential(
                    ocnn.modules.FcBnRelu(self.channels[self.min_lod], 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    nn.Linear(512, self.channel_latent * 2),
                )
                self.posterior_head = nn.Sequential(
                    ocnn.modules.FcBnRelu(posterior_channel_in, 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    ocnn.modules.FcBnRelu(512, 512),
                    nn.Linear(512, self.channel_latent * 2),
                )
                if self.use_full_layer:
                    self.global_pool = ocnn.nn.OctreeGlobalPool(nempty=True)
                else:
                    self.global_pool = ocnn.nn.OctreeGlobalPool(nempty=False)
                    self.global_pool_nempty = ocnn.nn.OctreeGlobalPool(nempty=True)
            else:
                self.prior_head = OctreeConvBnElu(
                    self.channels[self.min_lod], self.channel_latent * 2, nempty=False
                )
                self.posterior_head = OctreeConvBnElu(
                    posterior_channel_in, self.channel_latent * 2, nempty=False
                )

    def get_input_feature(self, octree, nempty=False, feature="F"):
        r"""Get the input feature from the input `octree`."""
        input_feature = ocnn.modules.InputFeature(feature, nempty=nempty)
        out = input_feature(octree)
        return out

    def get_ground_truth_signal(self, octree):
        input_feature = ocnn.modules.InputFeature("NF", nempty=True)
        data = input_feature(octree)
        return data

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
        return th.nn.Sequential(
            Conv1x1BnElu(channel_in, num_hidden),
            ocnn.modules.Conv1x1(num_hidden, channel_out, use_bias=True),
        )

    def encoder(
        self, octree, masks, depth, K, min_lod, max_lod, octree_z_min, frame_idx
    ):
        convs = dict()
        feat = self.get_input_feature(octree, feature=self.input_feature_type)

        convs[max_lod] = self.conv1(feat, octree, max_lod)
        for i, d in enumerate(range(max_lod, min_lod - 1, -1)):
            if d == min_lod and "M" in self.input_feature_type:
                with th.no_grad():
                    coords = get_xyz_from_octree(octree, d)
                    coords = octree_unnormalize_pts(
                        coords.clone(),
                        octree,
                        d,
                        octree_z_min,
                        self.grid_size,
                        self.grid_res,
                    )
                    coords = th.repeat_interleave(
                        coords, self.ofe_cubic_multiple, dim=0
                    ).reshape(-1, self.ofe_cubic_multiple, 3)
                    coords = (coords + self.ofe_offset).reshape(-1, 3)
                    batch_id = octree.batch_id(d).int()
                    batch_id = th.repeat_interleave(
                        batch_id, self.ofe_cubic_multiple, dim=0
                    )
                    octree_feature = self.ofe(
                        coords, masks, depth, K, batch_id, self.ofe_grid_size
                    ).float()
                if self.ofe_use_conv3d:
                    octree_feature = octree_feature.reshape(
                        -1,
                        self.ofe_quad_multiple,
                        self.ofe_quad_multiple,
                        self.ofe_quad_multiple,
                        2,
                    )
                    octree_feature = self.mask_head(
                        octree_feature.permute(0, 4, 1, 2, 3)
                    ).reshape(-1, self.ofe_out_channel)
                else:
                    octree_feature = octree_feature.reshape(
                        -1, self.ofe_cubic_multiple * 2
                    )
                    octree_feature = self.mask_head(octree_feature)
                convs[d] = th.cat([convs[d], octree_feature], dim=1)
            convs[d] = self.encoder_blks[i](convs[d], octree, d)
            if d > min_lod:
                convs[d - 1] = self.downsample[i](convs[d], octree, d)
        return convs

    def posterior_encoder(self, octree, min_lod, max_lod):
        feat = self.get_input_feature(octree, feature="NF")
        feat = self.posterior_conv1(feat, octree, max_lod)
        for i, d in enumerate(range(max_lod, min_lod - 1, -1)):
            feat = self.posterior_encoder_blks[i](feat, octree, d)
            if d > min_lod:
                feat = self.posterior_downsample[i](feat, octree, d)
        return feat

    def rsample(self, mu, var):
        """
        Return gaussian sample of (mu, var) using reparameterization trick.
        """
        eps = th.randn_like(mu)
        z = mu + eps * th.sqrt(var)
        return z

    def process_batch(self, batch, rgb_feat):
        _, masks, depth, pts_3d_in_list, pts_3d_gt_list, K, z_min, frame_idx = batch
        masks = th.cat([mask[0] for mask in masks], dim=0)

        B = depth.shape[0]
        device = depth.device

        octrees_in = []
        # octrees_scene_in = []
        octrees_in_scene_nnum = [0] * B
        octrees_mid = []
        octrees_out = []
        octrees_z_min = []
        num_objs = []

        min_lod = self.config.min_lod
        max_lod = self.config.max_lod
        img_height = self.config.img_height
        img_width = self.config.img_width

        for i, (pts_3d_in, pts_3d_gt) in enumerate(zip(pts_3d_in_list, pts_3d_gt_list)):
            #  Reconstruct an input octree
            obj_ids = th.unique(pts_3d_in.labels, sorted=True)
            num_objs.append(len(obj_ids))
            unnorm_pts_3d_in = unnormalize_pts(
                pts_3d_in.points.clone(), z_min[i], self.grid_size, self.grid_res
            )
            pts_2d_in = project_to_image_plane(
                unnorm_pts_3d_in, K[i], (img_height, img_width)
            )
            pc_features = F.grid_sample(
                rgb_feat[i : i + 1],
                pts_2d_in[None, None],
                align_corners=self.use_align_corners,
            )[0, :, 0]
            pts_3d_in.features = pc_features.transpose(0, 1)

            for oi in obj_ids:
                mask_in = (pts_3d_in.labels == oi).squeeze(-1)
                if self.use_full_layer:
                    octree_in = Octree(max_lod, min_lod, device=pts_3d_in.device)
                else:
                    octree_in = Octree(max_lod, min_lod - 1, device=pts_3d_in.device)
                masked_pts_3d_in = pts_3d_in.__getitem__(mask_in)
                octree_in.build_octree(masked_pts_3d_in)
                octrees_in_scene_nnum[i] += octree_in.nnum[min_lod].item()
                octrees_in.append(octree_in)

                #  Reconstruct a ground-truth octree
                if self.update_octree:
                    octree_out = Octree(max_lod, min_lod, device=pts_3d_in.device)
                    for d in range(min_lod + 1):
                        octree_out.octree_grow_full(depth=d)
                    octree_out.octree_grow(min_lod + 1)
                    octrees_out.append(octree_out)
                else:
                    mask_out = (pts_3d_gt.labels == oi).squeeze(-1)
                    masked_pts_3d_gt = pts_3d_gt.__getitem__(mask_out)

                    octree_out = Octree(max_lod, min_lod, device=pts_3d_in.device)
                    octree_out.build_octree(masked_pts_3d_gt)
                    octrees_out.append(octree_out)

                octrees_z_min.append(z_min[i].clone())

        octrees_in = ocnn.octree.merge_octrees(octrees_in)
        octrees_in.construct_all_neigh()
        # octrees_scene_in = ocnn.octree.merge_octrees(octrees_scene_in)
        # octrees_scene_in.construct_all_neigh()
        octrees_z_min = th.stack(octrees_z_min, dim=0)

        if self.update_octree:
            octrees_out = merge_octrees(octrees_out, depth=min_lod)
            for d in range(min_lod + 1):
                octrees_out.construct_neigh(d)
        else:
            octrees_out = ocnn.octree.merge_octrees(octrees_out)
            octrees_out.construct_all_neigh()

        num_objs = th.tensor(num_objs, device=pts_3d_in.device)
        depth = th.repeat_interleave(depth, num_objs, dim=0)

        batch = {
            "octrees_in": octrees_in,
            # 'octrees_scene_in': octrees_scene_in,
            "octrees_in_scene_nnum": octrees_in_scene_nnum,
            "octrees_out": octrees_out,
            "K": K,
            "depth": depth,
            "masks": masks,
            "octrees_z_min": octrees_z_min,
        }

        return batch

    def forward(self, batch):
        x = batch[0]
        frame_idx = batch[-1][0]
        if self.backbone is not None:
            x = self.backbone(x)

        batch = self.process_batch(batch, x)
        octrees_in = batch["octrees_in"]
        # octrees_scene_in = batch['octrees_scene_in']
        octrees_in_scene_nnum = batch["octrees_in_scene_nnum"]
        octrees_out = batch["octrees_out"]

        octrees_z_min = batch["octrees_z_min"]
        K = batch["K"][0]  # INFO(sh8): assuming that K is consistent over a batch
        masks = batch["masks"]
        depth = batch["depth"]

        # start = th.cuda.Event(enable_timing=True)
        # end = th.cuda.Event(enable_timing=True)
        # start.record()

        convs = self.encoder(
            octrees_in,
            masks,
            depth,
            K,
            self.min_lod,
            self.max_lod,
            octrees_z_min,
            frame_idx,
        )
        deconv = convs[self.min_lod]

        data = {"occs": [], "gt_occs": []}

        if self.use_cvae:
            if self.use_per_obj_latent:
                if self.use_full_layer:
                    prior_out = ocnn.nn.octree_depad(deconv, octrees_in, self.min_lod)
                else:
                    prior_out = deconv
                pooled_prior = self.global_pool(
                    prior_out, octrees_in, depth=self.min_lod
                )
                prior_out = self.prior_head(pooled_prior)
            else:
                prior_out = self.prior_head(deconv, octrees_in, depth=self.min_lod)

            pm = prior_out[:, : self.channel_latent]
            lpv = prior_out[:, self.channel_latent :]
            pv = th.exp(lpv)

            if self.update_octree:
                z = self.rsample(pm, pv)
                # z = pm
            else:
                posterior_out = self.posterior_encoder(
                    octrees_out, self.min_lod, self.max_lod
                )
                if self.use_pool_and_concat and self.use_per_obj_latent:
                    posterior_out = ocnn.nn.octree_depad(
                        posterior_out, octrees_out, self.min_lod
                    )
                    if self.use_full_layer:
                        posterior_out = self.global_pool(
                            posterior_out, octrees_out, depth=self.min_lod
                        )
                    else:
                        posterior_out = self.global_pool_nempty(
                            posterior_out, octrees_out, depth=self.min_lod
                        )
                    if self.concat_obj_latent:
                        posterior_out = th.cat([posterior_out, pooled_prior], dim=-1)
                    else:
                        posterior_out = posterior_out + pooled_prior
                    posterior_out = self.posterior_head(posterior_out)
                else:
                    posterior_out, _ = octree_align(
                        posterior_out,
                        octrees_out,
                        octrees_in,
                        self.min_lod,
                        nempty=False,
                    )
                    if self.concat_obj_latent:
                        posterior_out = th.cat([posterior_out, deconv], dim=-1)
                    else:
                        posterior_out = posterior_out + deconv
                    if self.use_per_obj_latent:
                        if self.use_full_layer:
                            posterior_out = ocnn.nn.octree_depad(
                                posterior_out, octrees_out, self.min_lod
                            )
                        posterior_out = self.global_pool(
                            posterior_out, octrees_out, depth=self.min_lod
                        )
                        posterior_out = self.posterior_head(posterior_out)
                    else:
                        posterior_out = self.posterior_head(
                            posterior_out, octrees_in, depth=self.min_lod
                        )
                qm = posterior_out[:, : self.channel_latent]
                lqv = posterior_out[:, self.channel_latent :]
                qv = th.exp(lqv)
                z = self.rsample(qm, qv)
                data["qm"] = qm
                data["qv"] = qv
                data["pm"] = pm
                data["pv"] = pv

            if self.use_per_obj_latent:
                batch_nnum = octrees_in.batch_nnum[self.min_lod].to(octrees_in.device)
                z = th.repeat_interleave(z, batch_nnum, dim=0)

            deconv = self.conv2(th.cat([deconv, z], dim=-1), octrees_in, self.min_lod)

        if self.use_mult_obj_enc:
            coords = get_xyz_from_octree(octrees_in, self.min_lod)
            deconv = self.mae_encoder(deconv, coords, octrees_in_scene_nnum)

        deconv, _ = octree_align(
            deconv, octrees_in, octrees_out, self.min_lod, nempty=False
        )
        for i, d in enumerate(range(self.min_lod, self.max_lod + 1)):
            if d > self.min_lod:
                deconv = self.upsample[i - 1](deconv, octrees_out, d - 1)
                skip = ocnn.nn.octree_align(
                    convs[d], octrees_in, octrees_out, d, nempty=False
                )
                deconv = deconv + skip

            deconv = self.decoder_blks[i](deconv, octrees_out, depth=d)
            output = self.predict[i](deconv)
            occ = output[:, :2]
            data["occs"].append(occ)

            if self.update_octree:
                split = occ.argmax(1).int()
                octrees_out.octree_split(split, d)
                if d < self.max_lod:
                    octrees_out.octree_grow(d + 1)
            else:
                gt_occ = octrees_out.nempty_mask(d).long()
                data["gt_occs"].append(gt_occ)

        data["signal"] = ocnn.nn.octree_depad(output[:, 2:], octrees_out, self.max_lod)
        if self.update_octree:
            octrees_out.normals[self.max_lod] = data["signal"][:, :3]
            octrees_out.features[self.max_lod] = data["signal"][:, 3:]
        else:
            data["gt_signal"] = self.get_ground_truth_signal(octrees_out)
        data["octrees_out"] = octrees_out

        return data
