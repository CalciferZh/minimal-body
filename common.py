class Skeleton:
  # The same as SMPL skeleton except that we remove the fake hand joints
  # and manually add a few keypoints
  n_keypoints = 33

  labels = [
    'pelvis', # 0
    'left_hip', 'right_hip', # 2
    'lowerback', # 3
    'left_knee', 'right_knee', # 5
    'upperback', # 6
    'left_ankle', 'right_ankle', # 8
    'thorax', # 9
    'left_toes', 'right_toes', # 11
    'lowerneck', # 12
    'left_clavicle', 'right_clavicle', # 14
    'upperneck', # 15
    'left_shoulder', 'right_shoulder', # 17
    'left_elbow', 'right_elbow', # 19
    'left_wrist', 'right_wrist', # 21
    # following are extended keypoints
    'head_top', 'left_eye', 'right_eye', # 24
    'left_hand_I0', 'left_hand_L0', # 26
    'right_hand_I0', 'right_hand_L0', # 28
    'left_foot_T0', 'left_foot_L0', # 30
    'right_foot_T0', 'right_foot_L0', # 32
  ]

  parents = [
    None,
    0, 0,
    0,
    1, 2,
    3,
    4, 5,
    6,
    7, 8,
    9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
    # extended
    15, 15, 15,
    20, 20,
    21, 21,
    7, 7,
    8, 8
  ]


HMAP_H = 64
HMAP_W = 48
IMG_H = 256
IMG_W = 192
