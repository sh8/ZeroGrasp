import os

import torch as th
import open3d as o3d
import numpy as np
import torch.nn.functional as F
from graspnetAPI import GraspGroup

from main import BaseTrainer
from zerograsp.utils.dataset import fetch_data
from zerograsp.utils.math import unnormalize_pts, rotation_6d_to_matrix
from zerograsp.utils.config import parse_config
from zerograsp.nets.utils import get_xyz_from_octree
from zerograsp.utils.collision_detector import ModelFreeCollisionDetector

GRASP_MAX_WIDTH = 0.1
GRASP_MAX_DEPTH = 0.04


def main():
    config = parse_config()
    config.update_octree = True

    print("Loading a model...")
    model = BaseTrainer.load_from_checkpoint(
        config.checkpoint, config=config, strict=False
    )
    model.cuda()
    model.eval()

    img_id = os.path.basename(config.img_path)[:3]

    print("Fetching data...")
    batch = fetch_data(
        config.img_path,
        config.depth_path,
        config.mask_path,
        config.camera_info_path,
        config,
        1.0,
    )
    grid_res = 1 << config.min_lod
    with th.no_grad():
        print("Running inference...")
        output = model.model(batch)
        z_min = batch[-2][0]
        pts_3d_in = batch[3][0]
        rays_3d = batch[4][0]

        octrees_out = output["octrees_out"]
        pcd, batch_id = get_xyz_from_octree(
            octrees_out, config.max_lod, nempty=True, return_batch=True
        )
        pcd = unnormalize_pts(pcd, z_min, config.grid_size, grid_res)
        normals = octrees_out.normals[config.max_lod]
        signal = octrees_out.features[config.max_lod]
        sdf = signal[:, :1]
        batch_id = batch_id.cpu().numpy()
        pcd = pcd.cpu().numpy()
        normals = F.normalize(normals, dim=-1).cpu().numpy()
        sdf = sdf.cpu().numpy()
        pcd = pcd - normals * sdf

        obj_ids = th.unique(pts_3d_in.labels, sorted=True)
        grasp_preds = []
        for i, oi in enumerate(obj_ids):
            mask = batch_id == i
            pcd_vis = o3d.geometry.PointCloud()
            pcd_vis.points = o3d.utility.Vector3dVector(pcd[mask])
            pcd_vis.normals = o3d.utility.Vector3dVector(normals[mask])
            pcd_vis.colors = o3d.utility.Vector3dVector(((normals[mask] + 1) / 2))
            new_pcd_path = f"demo/{img_id}_{i}.pts"
            o3d.io.write_point_cloud(new_pcd_path, pcd_vis)
            print("pcd is exported to", new_pcd_path)
            masked_pcd = pcd[mask]

            # Grasp Poses
            masked_signal = signal[mask, 1:]
            # quality = (masked_signal[valid, :1] * masked_signal[valid, 1:2]).cpu().numpy()
            # quality = (masked_signal[valid, :1] * masked_signal[valid, 1:2]).cpu().numpy()
            quality = masked_signal[:, :1].cpu().numpy()
            tangent = masked_signal[:, 2:5]
            gnormal = masked_signal[:, 5:8]
            R = rotation_6d_to_matrix(th.cat([-gnormal, tangent], dim=-1)).cpu().numpy()
            depth = masked_signal[:, 8:9].cpu().numpy()
            width = masked_signal[:, 9:10].cpu().numpy()
            translation = masked_pcd / 1000.0  # convert from mm to m
            height = 0.02 * np.ones_like(quality)
            grasp_preds = np.concatenate(
                [
                    quality,
                    np.clip(width * GRASP_MAX_WIDTH, 0.0, GRASP_MAX_WIDTH),
                    height,
                    np.clip(depth * GRASP_MAX_DEPTH, 0.0, GRASP_MAX_DEPTH),
                    R.reshape(-1, 9),
                    translation.reshape(-1, 3),
                    -1 * np.ones_like(quality),
                ],
                axis=-1,
            )

            gg = GraspGroup(grasp_preds).sort_by_score()
            depth_pcd = rays_3d.reshape(-1, 3)[::5].cpu().numpy()

            print("Number of grasps before collision detection", len(gg))

            with th.no_grad():
                cloud = th.from_numpy(pcd / 1000.0).float().cuda()
                cloud_nrm = th.from_numpy(normals).float().cuda()
                depth_cloud = th.from_numpy(depth_pcd / 1000.0).float().cuda()
                mfcdetector = ModelFreeCollisionDetector(cloud, cloud_nrm, depth_cloud)
                collision_mask, delta_width, refined_depth = mfcdetector.detect(gg)
                gg.grasp_group_array[:, 1] = gg.grasp_group_array[:, 1] + delta_width
                gg.grasp_group_array[:, 3] = refined_depth

            if (~collision_mask).sum() > 0:
                gg = gg[~collision_mask]

            gg = gg.nms(0.03, 30.0 / 180 * np.pi).sort_by_score()

            print("Number of grasps after collision detection", len(gg))
            gg.save_npy(f"demo/{img_id}_{i}.grasp.npy")
            print("grasp pose is exported")

        print("saved!")


if __name__ == "__main__":
    main()
