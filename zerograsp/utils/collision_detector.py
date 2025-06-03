""" Collision detection to remove collided grasp pose predictions.
Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
import open3d as o3d

GRASP_MAX_WIDTH = 0.1
GRASP_MAX_DEPTH = 0.04


def create_mesh_box(width, height, depth, num):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    vertices = th.zeros((num, 8, 3), device='cuda')
    vertices[:, 1::2, 0] = width
    vertices[:, 4:, 1] = height
    vertices[:, [2, 3, 6, 7], 2] = depth
    return vertices


def get_gripper_vertices(center, R, width, depth, table_trans=None):
    height = 0.02
    finger_width = 0.01
    depth_base = 0.02
    
    left_points = create_mesh_box(depth+depth_base+finger_width, finger_width, height, depth.shape[0])
    right_points = create_mesh_box(depth+depth_base+finger_width, finger_width, height, depth.shape[0])
    bottom_points = create_mesh_box(finger_width, width, height, depth.shape[0])

    left_points[..., 0] -= depth_base + finger_width
    left_points[..., 1] -= width/2 + finger_width
    left_points[..., 2] -= height/2

    right_points[..., 0] -= depth_base + finger_width
    right_points[..., 1] += width/2
    right_points[..., 2] -= height/2

    bottom_points[..., 0] -= finger_width + depth_base
    bottom_points[..., 1] -= width/2
    bottom_points[..., 2] -= height/2

    vertices = th.cat([left_points, right_points, bottom_points], dim=1)
    vertices = th.bmm(R, vertices.transpose(1, 2)).transpose(1, 2) + center.unsqueeze(1)

    if table_trans is not None:
        vertices = th.mm(table_trans[:3, :3], vertices.reshape(-1, 3).T).T.reshape(depth.shape[0], -1, 3) + table_trans[:3, 3].unsqueeze(0).unsqueeze(0)

    return vertices


class ModelFreeCollisionDetector():
    """ Collision detection in scenes without object labels. Current finger width and length are fixed.

        Input:
                scene_points: [torch.tensor, (N,3)]:
                    the scene points to detect

        Example usage:
            mfcdetector = ModelFreeCollisionDetector(scene_points)
            collision_mask = mfcdetector.detect(grasp_group, approach_dist=0.03)
            collision_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05, return_ious=True)
            collision_mask, empty_mask = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01)
            collision_mask, empty_mask, iou_list = mfcdetector.detect(grasp_group, approach_dist=0.03, collision_thresh=0.05,
                                            return_empty_grasp=True, empty_thresh=0.01, return_ious=True)
    """
    def __init__(self, pred_points, pred_normals, depth_points, use_collision_constraints=True, use_collision_detection_only_with_depth_map=False):
        self.finger_width = 0.01
        self.depth_base = 0.02
        self.batch_size = 500
        self.empty_thresh = 10
        self.depth_points = depth_points
        self.pred_points = pred_points
        self.pred_normals = pred_normals
        self.use_collision_constraints = use_collision_constraints
        self.use_collision_detection_only_with_depth_map = use_collision_detection_only_with_depth_map

    def detect(self, grasp_group, table_trans=None):
        """ Detect collision of grasps.

            Input:
                grasp_group: [GraspGroup, M grasps]
                    the grasps to check
                approach_dist: [float]
                    the distance for a gripper to move along approaching direction before grasping
                    this shifting space requires no point either
                collision_thresh: [float]
                    if global collision iou is greater than this threshold,
                    a collision is detected
                return_empty_grasp: [bool]
                    if True, return a mask to imply whether there are objects in a grasp
                empty_thresh: [float]
                    if inner space iou is smaller than this threshold,
                    a collision is detected
                    only set when [return_empty_grasp] is True
                return_ious: [bool]
                    if True, return global collision iou and part collision ious
                    
            Output:
                collision_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies collision
                [optional] empty_mask: [numpy.ndarray, (M,), numpy.bool]
                    True implies empty grasp
                    only returned when [return_empty_grasp] is True
                [optional] iou_list: list of [numpy.ndarray, (M,), numpy.float32]
                    global and part collision ious, containing
                    [global_iou, left_iou, right_iou, bottom_iou, shifting_iou]
                    only returned when [return_ious] is True
        """
        grasp_group_array = th.from_numpy(grasp_group.grasp_group_array).cuda()
        T = grasp_group_array[:, 13:16]
        R = grasp_group_array[:, 4:13].reshape((-1, 3, 3))
        heights = grasp_group_array[:, 2:3]
        depths = grasp_group_array[:, 3:4]
        widths = grasp_group_array[:, 1:2]

        num_grasps = T.shape[0]
        num_iter = num_grasps // self.batch_size + 1
        num_pred_points = self.pred_points.shape[0]
        collision_masks = []
        empty_masks = []

        antipodal_qualities = []
        delta_widths = []
        refined_depths = []
        # stabilities = []

        for i in range(num_iter):
            start = i * self.batch_size
            end = (i+1) * self.batch_size

            if self.use_collision_detection_only_with_depth_map:
                targets = self.depth_points.unsqueeze(0) - T[start:end].unsqueeze(1)
            else:
                targets = th.cat([self.pred_points, self.depth_points], dim=0).unsqueeze(0) - T[start:end].unsqueeze(1)

            targets = th.bmm(targets, R[start:end])

            ## collision detection
            # height mask
            mask1 = ((targets[:,:,2] > -heights[start:end]/2) & (targets[:,:,2] < heights[start:end]/2))
            # left finger mask
            mask2 = ((targets[:,:,0] > -self.depth_base) & (targets[:,:,0] < depths[start:end]))
            mask3 = (targets[:,:,1] > -(widths[start:end]/2 + self.finger_width))
            mask4 = (targets[:,:,1] < -widths[start:end]/2)
            # right finger mask
            mask5 = (targets[:,:,1] < (widths[start:end]/2 + self.finger_width))
            mask6 = (targets[:,:,1] > widths[start:end]/2)
            # bottom mask
            mask7 = (targets[:,:,0] < -self.depth_base) & (targets[:,:,0] > -(self.depth_base + self.finger_width))
            # get collision mask of each point
            left_mask = (mask1 & mask2 & mask3 & mask4)
            right_mask = (mask1 & mask2 & mask5 & mask6)
            bottom_mask = (mask1 & mask3 & mask5 & mask7)
            inner_mask = (mask1 & mask2 & (~mask4) & (~mask6))
            # collision_mask = th.any((left_mask | right_mask | bottom_mask), dim=-1)
            # collision_mask = th.sum((left_mask | right_mask | bottom_mask), dim=-1) > 0
            # print('num of collisions', th.sum((left_mask | right_mask | bottom_mask), dim=-1))
            # print('num of collisions', th.any((left_mask | right_mask | bottom_mask), dim=-1))
            # print('empty_mask', th.sum(inner_mask, dim=-1))
            empty_mask = th.sum(inner_mask, dim=-1) < self.empty_thresh
            empty_masks.append(empty_mask)

            # contact points
            contacts = targets.clone()
            contacts[..., 1] = contacts[..., 1] * inner_mask
            contact_dist_left, contact_idx_left = th.min(widths[start:end]/2 + contacts[:, :, 1], dim=1)
            contact_dist_right, contact_idx_right = th.min(widths[start:end]/2 - contacts[:, :, 1], dim=1)

            contact_dist = th.minimum(contact_dist_right, contact_dist_left)
            delta_width = 2.0 * (th.clamp(contact_dist, min=0.0025, max=0.01) - contact_dist)
            if self.use_collision_constraints:
                delta_widths.append(delta_width)
            else:
                delta_widths.append(delta_width * 0.0)

            contact_idx_left = contact_idx_left.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
            contact_idx_right = contact_idx_right.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
            contact_left = th.gather(contacts, 1, contact_idx_left)[:, 0] # (N_g, 3)
            contact_right = th.gather(contacts, 1, contact_idx_right)[:, 0] # (N_g, 3)
            contact_depth = th.clamp(th.minimum(th.maximum(contact_left[:, 0], contact_right[:, 0]), depths[start:end, 0]), min=0.0)
            if self.use_collision_constraints:
                refined_depths.append(contact_depth)
            else:
                refined_depths.append(depths[start:end, 0])

            # recompute a collision mask
            mask2 = ((targets[:,:,0] > -self.depth_base) & (targets[:,:,0] < contact_depth.unsqueeze(-1)))
            left_mask = (mask1 & mask2 & mask3 & mask4)
            right_mask = (mask1 & mask2 & mask5 & mask6)
            collision_mask = th.any((left_mask | right_mask | bottom_mask), dim=-1)
            collision_masks.append(collision_mask)

        empty_masks = th.cat(empty_masks, dim=0)
        delta_widths = th.cat(delta_widths, dim=0)
        refined_depths = th.cat(refined_depths, dim=0)
        collision_masks = th.cat(collision_masks, dim=0)
        collision_masks = (collision_masks | empty_masks)

        if table_trans is not None:
            vertices = get_gripper_vertices(T, R, widths + delta_widths.unsqueeze(-1), refined_depths.unsqueeze(-1), table_trans)
            table_collision_masks = th.any(vertices[:, :, 2] < 0.0, dim=-1)
            collision_masks = (collision_masks | table_collision_masks)

        return collision_masks.cpu().numpy(), delta_widths.cpu().numpy(), refined_depths.cpu().numpy()
