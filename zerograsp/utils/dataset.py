import io
import random
import json
import yaml

import cv2
import numpy as np
import torch as th
from PIL import Image
import imageio.v3 as iio
from torchvision import transforms
from ocnn.octree import Points

from zerograsp.utils.math import get_camera_rays, rle_to_binary_mask, normalize_pts


def decode_depth(key, data):
    if not (key.endswith("depth.png") or key.endswith("depth_st.png")):
        return None
    return np.asarray(iio.imread(io.BytesIO(data)), dtype=np.float32)


def make_sample_wrapper(
    config,
    is_eval=False,
    K=[
        [572.41136339, 0.0, 325.2611084],
        [0.0, 573.57043286, 242.04899588],
        [0.0, 0.0, 1.0],
    ],
):
    img_size = (config.img_height, config.img_width)  # should use a config
    should_resize_square = config.backbone_model is not None and (
        "dinov2" in config.backbone_model or "clip" in config.backbone_model
    )
    resized_img_size = (224, 224) if should_resize_square else (480, 640)
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    K = np.asarray(K, dtype=np.float32)  # should use a config
    camera_rays = get_camera_rays(K, img_size)

    transform = transforms.Compose(
        [
            transforms.Resize(resized_img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    def make_sample(sample):
        rgb = sample["rgb.jpg"].crop(
            (0, 0, config.img_width, config.img_height)
        )  # Crop a stereo image
        if is_eval or config.use_gt_depth:
            depth = sample["camera.json"]["depth_scale"] * sample["depth.png"]
        else:
            depth = sample["depth_st.png"]
        depth = depth.astype(np.float32)
        if config.train_dataset_name == "mirage" and config.use_gt_depth:
            depth += (0.5 + np.maximum((depth - 500.0) / 1000.0, 0)) * np.random.normal(
                size=depth.shape
            )
        rgb = transform(rgb)
        obj_pose = sample["gt.json"]
        obj_info = sample["gt_info.json"]
        mask_rle = sample["mask_visib.json"]
        K = np.asarray(sample["camera.json"]["cam_K"]).astype(np.float32).reshape(3, 3)
        spc = th.from_numpy(sample["spc.npz"]["spc"].astype(np.float32))
        obj_ids = th.from_numpy(sample["spc.npz"]["obj_ids"].astype(np.int32))
        if not is_eval:
            grasp_poses = th.from_numpy(
                sample["grasp.npz"]["grasp_poses"].astype(np.float32)
            )

        masks = []
        dilated_masks = []
        visible_pts_3d = []
        visible_labels = []

        rays_pts_3d = th.from_numpy(
            (camera_rays * depth[:, :, None]).astype(np.float32)
        )

        filtered_idxs = []
        idxs = np.argsort(np.array([op["obj_id"] for op in obj_pose])).tolist()
        tmp = sorted(zip(obj_info, obj_pose), key=lambda x: x[1]["obj_id"])
        obj_info, obj_pose = map(list, zip(*tmp))
        for idx, oi, op in zip(idxs, obj_info, obj_pose):
            if is_eval:
                visib_fract_thresh = 0.0
            else:
                visib_fract_thresh = 0.2

            if (
                oi["visib_fract"] <= visib_fract_thresh
            ):  # get rid of heavily occluded objects
                continue

            imask_rle = mask_rle[str(idx)]
            imask = rle_to_binary_mask(imask_rle, oi["bbox_visib"])

            float_imask = imask.astype(np.float32)
            if not is_eval:
                flag = bool(random.getrandbits(1))
                if flag:
                    kernel_size = random.choice([1, 3, 5])
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    float_imask = cv2.dilate(float_imask, kernel, iterations=1)
                else:
                    shift_x = random.uniform(-3, 3)
                    shift_y = random.uniform(-3, 3)
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    float_imask = cv2.warpAffine(
                        float_imask, M, (img_size[1], img_size[0])
                    )

            imask = float_imask > 0.5
            imask = np.logical_and(imask, depth > 10.0)
            if imask.sum() < 10:
                # print('imask sum', imask.sum(), op['obj_id'])
                continue
            masks.append(imask)

            kernel = np.ones((5, 5), np.uint8)
            dilated_imask = cv2.dilate(float_imask, kernel, iterations=1) > 0.5
            dilated_masks.append(dilated_imask)
            filtered_idxs.append(op["obj_id"])

            masked_pts_3d = rays_pts_3d[imask > 0.0].reshape(-1, 3)
            # masked_pts_3d_normals = pts_3d_normals[imask > 0.0].reshape(-1, 3)
            if config.train_dataset_name == "graspnet":
                masked_pts_3d = masked_pts_3d[::2]
                # masked_pts_3d_normals = masked_pts_3d_normals[::2]
            # print('mask num', masked_pts_3d.shape[0], op['obj_id'])
            masked_labels = (
                th.ones((masked_pts_3d.shape[0], 1), dtype=th.long) * op["obj_id"]
            )
            visible_pts_3d.append(masked_pts_3d)
            visible_labels.append(masked_labels)
            # print('obj_id:', op['obj_id'], 'px_count_visib', oi['px_count_valid'], 'maskd_pts num', masked_pts_3d.shape[0], 'visib_fract:', oi['visib_fract'])

        visible_pts_3d = th.cat(visible_pts_3d, dim=0)
        visible_labels = th.cat(visible_labels, dim=0)
        z_min = (th.min(visible_pts_3d[:, 2]) // grid_size) * grid_size - 5 * grid_size

        if config.model_name == "octmae":
            masks = np.any(np.stack(masks, axis=-1), axis=-1)
        else:
            masks = np.stack(masks)
            dilated_masks = np.stack(dilated_masks)
            flat_masks = masks.reshape(masks.shape[0], -1)
            flat_dilated_masks = dilated_masks.reshape(dilated_masks.shape[0], -1)
            overlap_matrix = np.dot(flat_dilated_masks, flat_dilated_masks.T)
            np.fill_diagonal(overlap_matrix, False)
            neighbor_masks = overlap_matrix @ flat_masks
            neighbor_masks = neighbor_masks.reshape(masks.shape[0], *img_size)
            masks = [np.stack([masks, neighbor_masks], axis=-1)]

        vdb_grid_size = 2.5
        offset = vdb_grid_size * 0.5  # this offset is to fix the misalignment in VDB

        spc_mask = th.logical_and(spc[:, 3] < vdb_grid_size, spc[:, 3] > -vdb_grid_size)
        spc = spc[spc_mask]
        obj_ids = obj_ids[spc_mask]

        if is_eval:
            features = spc[:, 3:4]
        else:
            grasp_poses = grasp_poses[spc_mask]
            if grasp_poses.shape[1] == 9:
                grasp_poses = th.cat(
                    [grasp_poses, th.zeros(grasp_poses.shape[0], 1)], dim=-1
                )
            features = th.cat([spc[:, 3:4], grasp_poses], dim=-1)
            features = th.nan_to_num(features, posinf=0.0, neginf=0.0)

        pts_3d_in = Points(
            points=normalize_pts(visible_pts_3d, z_min, grid_size, grid_res),
            labels=visible_labels,
        )
        pts_3d_gt = Points(
            points=normalize_pts(spc[:, :3] + offset, z_min, grid_size, grid_res),
            normals=spc[:, 4:7],
            features=features,
            labels=obj_ids,
        )

        if config.use_aug and (not is_eval):
            tangential = pts_3d_gt.features[:, 3:6]
            normal = pts_3d_gt.features[:, 6:9]
            axis_map = {"x": 0, "y": 1, "z": 2}
            for axis in "xy":
                flag = bool(random.getrandbits(1))
                if flag:
                    pts_3d_in.flip(axis)
                    pts_3d_gt.flip(axis)
                    tangential[:, axis_map[axis]] = -tangential[:, axis_map[axis]]
                    normal[:, axis_map[axis]] = -normal[:, axis_map[axis]]
            angle = th.tensor([0.0, 0.0, (random.random() * np.pi / 3) - np.pi / 6])
            pts_3d_in.rotate(angle)
            pts_3d_gt.rotate(angle)
            cos, sin = angle.cos(), angle.sin()
            rotz = th.Tensor([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])
            tangential = tangential @ rotz
            normal = normal @ rotz
            pts_3d_gt.features[:, 3:6] = tangential
            pts_3d_gt.features[:, 6:9] = normal

        if pts_3d_in.points.shape[0] < 100 or pts_3d_gt.points.shape[0] < 1000:
            frame_idx = sample["__key__"]
            raise Exception("This item does not have enough points", frame_idx)

        pts_3d_in.clip()
        pts_3d_gt.clip()

        # unique_labels = th.unique(pts_3d_in.labels)
        # filtered_idxs = np.array(filtered_idxs)
        # if not np.array_equal(unique_labels.numpy(), filtered_idxs):
        #     print(unique_labels.numpy(), filtered_idxs)
        #     raise Exception('The number of input labels is different from the number of masks')

        return (rgb, masks, depth, pts_3d_in, pts_3d_gt, K, z_min, sample["__key__"])

    return make_sample


def fetch_data(
    rgb_path, depth_path, mask_path, camera_path, config, depth_scale=1.0, device="cuda"
):
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    img_size = (config.img_height, config.img_width)  # should use a config
    should_resize_square = config.backbone_model is not None and (
        "dinov2" in config.backbone_model or "clip" in config.backbone_model
    )
    resized_img_size = (
        (224, 224) if should_resize_square else (config.img_height, config.img_width)
    )
    transform = transforms.Compose(
        [
            transforms.Resize(resized_img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    rgb = transform(Image.open(rgb_path))
    depth = np.asarray(iio.imread(depth_path), dtype=np.float32)
    depth = depth_scale * depth

    mask = np.asarray(Image.open(mask_path))
    obj_ids = np.unique(mask)[1:]

    if camera_path.endswith(".json"):
        with open(camera_path, "r") as f:
            K = np.array(json.load(f)["cam_K"]).astype(np.float32).reshape(3, 3)
    if camera_path.endswith(".yml"):
        with open(camera_path, "r") as f:
            K = (
                extract_camera_matrix(yaml.safe_load(f))
                .astype(np.float32)
                .reshape(3, 3)
            )
    else:
        raise Exception("We do not support this format")
    camera_rays = get_camera_rays(K, img_size)
    rays_pts_3d = th.from_numpy((camera_rays * depth[:, :, None]).astype(np.float32))

    masks = []
    dilated_masks = []
    visible_pts_3d = []
    visible_labels = []
    for obj_id in obj_ids:
        imask = mask == obj_id
        float_imask = imask.astype(np.float32)

        imask = float_imask > 0.5
        imask = np.logical_and(imask, depth > 10.0)
        masks.append(imask)

        kernel = np.ones((5, 5), np.uint8)
        dilated_imask = cv2.dilate(float_imask, kernel, iterations=1) > 0.5
        dilated_masks.append(dilated_imask)

        masked_pts_3d = rays_pts_3d[imask > 0.0].reshape(-1, 3)
        masked_labels = th.ones((masked_pts_3d.shape[0], 1), dtype=th.long) * obj_id
        visible_pts_3d.append(masked_pts_3d)
        visible_labels.append(masked_labels)

    visible_pts_3d = th.cat(visible_pts_3d, dim=0)
    visible_labels = th.cat(visible_labels, dim=0)

    z_min = (th.min(visible_pts_3d[:, 2]) // grid_size) * grid_size - 5 * grid_size
    pts_3d_in = Points(
        normalize_pts(visible_pts_3d, z_min, grid_size, grid_res), labels=visible_labels
    )
    pts_3d_in.clip()

    masks = np.stack(masks)
    dilated_masks = np.stack(dilated_masks)
    flat_masks = masks.reshape(masks.shape[0], -1)
    flat_dilated_masks = dilated_masks.reshape(dilated_masks.shape[0], -1)
    overlap_matrix = np.dot(flat_dilated_masks, flat_dilated_masks.T)
    np.fill_diagonal(overlap_matrix, False)
    neighbor_masks = overlap_matrix @ flat_masks
    neighbor_masks = neighbor_masks.reshape(masks.shape[0], *img_size)
    masks = np.stack([masks, neighbor_masks], axis=-1)[None]

    # Transfer data to a GPU
    rgb = rgb.to(device)
    masks = th.from_numpy(masks).to(device)
    depth = th.from_numpy(depth).to(device)
    pts_3d_in = pts_3d_in.to(device)
    K = th.from_numpy(K).to(device)
    z_min = z_min.to(device)

    return (
        rgb[None],
        masks[None],
        depth[None],
        [pts_3d_in],
        [rays_pts_3d],
        K[None],
        z_min[None],
        [rgb_path],
    )


def extract_camera_matrix(meta, proj: bool = True, right: bool = False):
    # Get name of the matrix
    name = ("right_" if right else "left_") + ("p" if proj else "k")

    # Extract matrix (remove translation vector for projection matrix)
    if proj:
        matrix = np.array(meta[name]).reshape(3, 4)[:3, :3]
    else:
        matrix = np.array(meta[name]).reshape(3, 3)

    # Apply scale factor on projection matrix
    if "rectified_width" in meta:
        if proj and meta["width"] and meta["rectified_width"]:
            scale = meta["rectified_width"] / meta["width"]
            matrix[:2, :] /= scale

    return matrix
