import torch as th
import torch.nn.functional as F
import numpy as np


def rle_to_binary_mask(rle, bbox_visib=None):
    """Converts a COCOs run-length encoding (RLE) to binary mask.

    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get("size")), dtype=bool)
    counts = rle.get("counts")

    start = 0

    if (
        bbox_visib is not None
        and len(counts) % 2 == 0
        and bbox_visib[0] == 0
        and bbox_visib[1] == 0
    ):
        counts.insert(0, 0)

    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get("size"), order="F")

    return binary_mask


def get_camera_rays(K, img_size, inv_scale=1):
    u, v = np.meshgrid(
        np.arange(0, img_size[1], inv_scale), np.arange(0, img_size[0], inv_scale)
    )

    # Convert to homogeneous coordinates
    u = u.reshape(-1)
    v = v.reshape(-1)
    ones = np.ones(u.shape[0])
    uv1 = np.stack((u, v, ones), axis=-1)  # shape (H*W, 3)

    K_inv = np.linalg.inv(K)
    pts = np.dot(uv1, K_inv.T)  # shape (H*W, 3)
    pts = pts.reshape((img_size[0] // inv_scale, img_size[1] // inv_scale, 3))

    return pts


def normalize_pts(pts, z_min, grid_size, grid_res):
    pts[:, :2] = pts[:, :2] / (grid_res * grid_size // 2)  # (-1, 1)
    pts[:, 2] = (((pts[:, 2] - z_min) / (grid_res * grid_size)) - 0.5) * 2.0  # (-1, 1)
    return pts


def unnormalize_pts(pts, z_min, grid_size, grid_res):
    pts[:, :2] = (grid_res * grid_size // 2) * pts[:, :2]
    pts[:, 2] = (0.5 * pts[:, 2] + 0.5) * grid_res * grid_size + z_min
    return pts


def project_to_image_plane(pts_3d, K, img_size):
    pts_2d_homogeneous = pts_3d @ K.T  # shape (N, 3)
    pts_2d = pts_2d_homogeneous[:, :2] / (pts_2d_homogeneous[:, 2:3] + 1e-4)
    pts_2d[:, 0] /= img_size[1]
    pts_2d[:, 1] /= img_size[0]
    pts_2d = (pts_2d - 0.5) * 2.0  # (-1, 1)
    return pts_2d


def rotation_6d_to_matrix(d6: th.Tensor) -> th.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = th.cross(b1, b2, dim=-1)
    return th.stack((b1, b2, b3), dim=-1)
