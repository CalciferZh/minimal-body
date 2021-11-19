import torch
from torch import nn
import utils
import numpy as np
import cv2
import common as cm


class ConvBN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, padding=0):
    super(ConvBN, self).__init__()
    self.conv = nn.Conv2d(
      in_channels, out_channels, kernel_size,
      stride=stride, dilation=dilation, padding=padding, bias=False
    )
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, layer):
    layer = self.conv(layer)
    layer = self.bn(layer)
    return layer


class ConvBNReLU(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0):
    super(ConvBNReLU, self).__init__()
    self.conv_bn = ConvBN(
      in_channels, out_channels, kernel_size,
      stride=stride, dilation=dilation, padding=padding
    )
    self.relu = nn.ReLU()

  def forward(self, layer):
    layer = self.conv_bn(layer)
    layer = self.relu(layer)
    return layer


class BottleNeck(nn.Module):
  def __init__(self, in_channels, out_channels, stride, dilation=1):
    super(BottleNeck, self).__init__()

    self.downsample = None
    if in_channels == out_channels:
      if stride != 1:
        self.downsample = nn.MaxPool2d(kernel_size=stride, stride=stride)
    else:
      self.downsample = ConvBN(in_channels, out_channels, 1, stride)

    hidden_width = out_channels // 4
    self.op_1 = ConvBNReLU(in_channels, hidden_width, 1, 1)
    self.op_2 = ConvBNReLU(
      hidden_width, hidden_width, 3, stride, dilation, padding=dilation
    )
    self.op_3 = ConvBN(hidden_width, out_channels, 1, 1)
    self.relu = nn.ReLU()

  def forward(self, layer):
    if self.downsample is None:
      shortcut = layer
    else:
      shortcut = self.downsample(layer)

    layer = self.op_1(layer)
    layer = self.op_2(layer)
    layer = self.op_3(layer)

    layer = layer + shortcut
    layer = self.relu(layer)

    return layer


class ResNet50(nn.Module):
  def __init__(self, size=[3, 4, 6]):
    super(ResNet50, self).__init__()

    operations = [ConvBNReLU(3, 64, 7, 2, padding=3)] + \
                 [BottleNeck(64, 256, 1)] + \
                 [BottleNeck(256, 256, 1) for _ in range(size[0] - 2)] + \
                 [BottleNeck(256, 256, 2)] + \
                 [BottleNeck(256, 512, 1, 2)] + \
                 [BottleNeck(512, 512, 1, 2) for _ in range(size[1] - 1)] + \
                 [BottleNeck(512, 1024, 1, 4)] + \
                 [BottleNeck(1024, 1024, 1, 4) for _ in range(size[2] - 1)]
    self.operation = nn.Sequential(*operations)

  def forward(self, layer):
    layer = self.operation(layer)
    return layer


class RegressionNet2D(nn.Module):
  def __init__(self, in_channels, n_keypoints, hidden_width=256):
    super(RegressionNet2D, self).__init__()
    self.projection = ConvBNReLU(in_channels, hidden_width, 3, padding=1)
    self.prediction = nn.Conv2d(hidden_width, n_keypoints, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, layer):
    layer = self.projection(layer)
    layer = self.prediction(layer)
    layer = self.sigmoid(layer)
    return layer


class RegressionNet3D(nn.Module):
  def __init__(self, in_channels, n_keypoints, hidden_width=256):
    super(RegressionNet3D, self).__init__()
    self.projection = ConvBNReLU(in_channels, hidden_width, 3, padding=1)
    self.prediction = nn.Conv2d(hidden_width, n_keypoints * 3, 1)

  def forward(self, layer):
    layer = self.projection(layer)
    layer = self.prediction(layer)
    return layer


class PoseNet3D(nn.Module):
  def __init__(self, in_channels, n_keypoints):
    super(PoseNet3D, self).__init__()
    self.hmap_regressor = RegressionNet2D(in_channels, n_keypoints)
    in_channels += n_keypoints
    self.lmap_regressor = RegressionNet3D(in_channels, n_keypoints)

  def forward(self, features):
    hmap = self.hmap_regressor(features)
    features = torch.cat([features, hmap], 1)
    lmap = self.lmap_regressor(features)
    return {'hmap': hmap, 'lmap': lmap}


class MinimalBody(nn.Module):
  def __init__(self, device):
    super(MinimalBody, self).__init__()
    self.xyz_indices = torch.arange(0, cm.Skeleton.n_keypoints, device=device) * cm.HMAP_H * cm.HMAP_W
    self.feature_extactor = ResNet50()
    self.humbi_posenet = PoseNet3D(1024, cm.Skeleton.n_keypoints)
    self.device = device

  def forward(self, image):
    image = cv2.resize(image, (cm.IMG_W, cm.IMG_H), cv2.INTER_LINEAR)
    image = np.transpose(np.expand_dims(image, 0), [0, 3, 1, 2])
    image = torch.from_numpy(image).to(self.device).float() / 255

    features = self.feature_extactor(image)
    humbi_kpts = self.humbi_posenet(features)

    uv = utils.hmap_to_uv(humbi_kpts['hmap'][0])
    lmap = torch.reshape(humbi_kpts['lmap'][0], [-1, 3, cm.HMAP_H, cm.HMAP_W])
    xyz = utils.lmap_uv_to_xyz(uv, lmap, self.xyz_indices)

    return uv.detach().cpu().numpy(), xyz.detach().cpu().numpy()
