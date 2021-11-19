import torch
import utils
import network
import numpy as np
import imageio
import common as cm


test_img_path = 'test_input.jpg'
device = 'cuda'


model = network.MinimalBody(device)
model.load_state_dict(torch.load('model/minimal_body_v1.pth'))
model.to(device)
model.eval()

# need to be 4:3 portrait
img = imageio.imread(test_img_path)

with torch.no_grad():
  uv, xyz = model(img)

canvas = imageio.imread(test_img_path)
scale = canvas.shape[0] / cm.HMAP_H
utils.render_bones_from_uv(
  np.round(uv * scale).astype(np.int), canvas, cm.Skeleton.parents,
  color=utils.get_bone_color(cm.Skeleton.labels)
)
imageio.imsave('test_output_uv.jpg', canvas)

v, f = utils.joints_to_mesh(xyz, cm.Skeleton.parents)
utils.obj_save('test_output_xyz.obj', v, f)
