# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.utils.checkpoint
from typing import List

from ocnn.octree import Octree
from ocnn.nn import OctreeMaxPool, OctreeConv, OctreeDeconv
from ocnn.modules import Conv1x1Bn, Conv1x1

bn_momentum, bn_eps = 0.01, 0.001


class OctreeResBlock(torch.nn.Module):
  r''' Octree-based ResNet block in a bottleneck style. The block is composed of
  a series of :obj:`Conv1x1`, :obj:`Conv3x3`, and :obj:`Conv1x1`.

  Args:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    stride (int): The stride of the block (:obj:`1` or :obj:`2`).
    bottleneck (int): The input and output channels of the :obj:`Conv3x3` is
        equal to the input channel divided by :attr:`bottleneck`.
    nempty (bool): If True, only performs the convolution on non-empty
        octree nodes.
  '''

  def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
               bottleneck: int = 4, nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bottleneck = bottleneck
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.max_pool = OctreeMaxPool(nempty)
    self.conv1x1a = Conv1x1BnElu(in_channels, channelb)
    self.conv3x3 = OctreeConvBnElu(channelb, channelb, nempty=nempty)
    self.conv1x1b = Conv1x1Bn(channelb, out_channels)
    if self.in_channels != self.out_channels:
      self.conv1x1c = Conv1x1Bn(in_channels, out_channels)
    self.elu = torch.nn.GELU()
    self.elu = torch.nn.ELU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.max_pool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv1x1a(data)
    conv2 = self.conv3x3(conv1, octree, depth)
    conv3 = self.conv1x1b(conv2)
    if self.in_channels != self.out_channels:
      data = self.conv1x1c(data)
    out = self.elu(conv3 + data)
    return out


class OctreeResBlocks(torch.nn.Module):
  r''' A sequence of :attr:`resblk_num` ResNet blocks.
  '''

  def __init__(self, in_channels, out_channels, resblk_num, bottleneck=4,
               nempty=False, resblk=OctreeResBlock, use_checkpoint=False):
    super().__init__()
    self.resblk_num = resblk_num
    self.use_checkpoint = use_checkpoint
    channels = [in_channels] + [out_channels] * resblk_num

    self.resblks = torch.nn.ModuleList(
        [resblk(channels[i], channels[i+1], 1, bottleneck, nempty)
         for i in range(self.resblk_num)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    for i in range(self.resblk_num):
      if self.use_checkpoint:
        data = torch.utils.checkpoint.checkpoint(
            self.resblks[i], data, octree, depth)
      else:
        data = self.resblks[i](data, octree, depth)
    return data


class OctreeConvBnElu(torch.nn.Module):
  r''' A sequence of :class:`OctreeConv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeConv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.elu = torch.nn.GELU()

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.bn(out)
    out = self.elu(out)
    return out


class OctreeDeconvBnElu(torch.nn.Module):
  r''' A sequence of :class:`OctreeDeconv`, :obj:`BatchNorm`, and :obj:`Relu`.

  Please refer to :class:`ocnn.nn.OctreeDeconv` for details on the parameters.
  '''

  def __init__(self, in_channels: int, out_channels: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.elu = torch.nn.GELU()

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    out = self.bn(out)
    out = self.elu(out)
    return out


class Conv1x1BnElu(torch.nn.Module):
  r''' A sequence of :class:`Conv1x1`, :class:`BatchNorm` and :class:`Relu`.
  '''

  def __init__(self, in_channels: int, out_channels: int):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.bn = torch.nn.BatchNorm1d(out_channels, bn_eps, bn_momentum)
    self.elu = torch.nn.GELU()

  def forward(self, data: torch.Tensor):
    r''''''

    out = self.conv(data)
    out = self.bn(out)
    out = self.elu(out)
    return out
