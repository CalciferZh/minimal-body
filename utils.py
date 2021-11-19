import numpy as np
import cv2
import torch
from transforms3d.axangles import axangle2mat


def render_bones_from_uv(uv, canvas, parents, color):
  thickness = int(max(round(canvas.shape[0] / 128), 1))
  for child, parent in enumerate(parents):
    if parent is None:
      continue
    c = color[child]
    start = (int(uv[parent][1]), int(uv[parent][0]))
    end = (int(uv[child][1]), int(uv[child][0]))
    cv2.line(canvas, start, end, c, thickness)
  return canvas


def obj_save(path, vertices, faces):
  with open(path, 'w') as fp:
    for v in vertices:
      v = np.ravel(v)
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    for f in faces + 1:
      f = np.ravel(f)
      fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def get_bone_color(labels):
  color = []
  for l in labels:
    if 'left' in l:
      color.append([255, 0, 0])
    elif 'right' in l:
      color.append([0, 255, 0])
    else:
      color.append([0, 0, 255])
  return color


def hmap_to_uv(hmap):
  hmap_flat = torch.reshape(hmap, (hmap.shape[0], -1))
  argmax = torch.argmax(hmap_flat, axis=-1)
  argmax_r = argmax // hmap.shape[2]
  argmax_c = argmax % hmap.shape[2]
  uv = torch.stack([argmax_r, argmax_c], axis=1)
  return uv


def lmap_uv_to_xyz(uv, lmap, offset, w=48):
  xyz = torch.stack(
    [
      torch.take(lmap[:, 0], offset + uv[:, 0] * w + uv[:, 1]),
      torch.take(lmap[:, 1], offset + uv[:, 0] * w + uv[:, 1]),
      torch.take(lmap[:, 2], offset + uv[:, 0] * w + uv[:, 1]),
    ], -1
  )
  return xyz


def joints_to_mesh(joints, parents, thickness=0.2):
  n_bones = len(list(filter(lambda x: x is not None, parents)))
  faces = np.empty([n_bones * 8, 3], dtype=np.int32)
  verts = np.empty([n_bones * 6, 3], dtype=np.float32)
  bone_idx = -1
  for child, parent in enumerate(parents):
    if parent is None:
      continue
    bone_idx += 1
    a = joints[parent]
    b = joints[child]
    ab = b - a
    f = a + thickness * ab

    if ab[0] == 0:
      ax = [0, 1, 0]
    else:
      ax = [-ab[1]/ab[0], 1, 0]

    fd = np.transpose(axangle2mat(ax, -np.pi/2).dot(np.transpose(ab))) \
         * thickness / 1.2
    d = fd + f
    c = np.transpose(axangle2mat(ab, -np.pi/2).dot(np.transpose(fd))) + f
    e = np.transpose(axangle2mat(ab, np.pi/2).dot(np.transpose(fd))) + f
    g = np.transpose(axangle2mat(ab, np.pi).dot(np.transpose(fd))) + f

    verts[bone_idx*6+0] = a
    verts[bone_idx*6+1] = b
    verts[bone_idx*6+2] = c
    verts[bone_idx*6+3] = d
    verts[bone_idx*6+4] = e
    verts[bone_idx*6+5] = g

    faces[bone_idx*8+0] = \
      np.flip(np.array([0, 2, 3], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+1] = \
      np.flip(np.array([0, 3, 4], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+2] = \
      np.flip(np.array([0, 4, 5], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+3] = \
      np.flip(np.array([0, 5, 2], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+4] = \
      np.flip(np.array([1, 4, 3], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+5] = \
      np.flip(np.array([1, 3, 2], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+6] = \
      np.flip(np.array([1, 5, 4], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+7] = \
      np.flip(np.array([1, 2, 5], dtype=np.int32), axis=0) + bone_idx * 6

  return verts, faces
