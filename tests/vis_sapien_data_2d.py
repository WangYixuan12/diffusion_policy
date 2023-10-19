#!/usr/bin/env python
# coding: utf-8

import h5py
import numpy as np
import cv2

import sys
sys.path.append('/home/yixuan/diffusion_policy')

from diffusion_policy.common.data_utils import load_dict_from_hdf5

num_episodes = 1

for i in range(num_episodes):
    print(f'episode {i}')
    data_path = f'/media/yixuan_2T/diffusion_policy/data/sapien_env/teleop_data/pick_place_soda/2023-10-18-15-06-10-582709/episode_{i}.hdf5'

    data_dict = load_dict_from_hdf5(data_path)

    obs_key = 'right_view_color'

    obs = data_dict['observations']['images'][obs_key]

    T = obs.shape[0]

    for t in range(T):
        img = obs[t]
        cv2.imshow('img', img)
        cv2.waitKey(10)
        if t == 0:
            vid = cv2.VideoWriter(f'episode_{i}_right.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (img.shape[1], img.shape[0]))
        vid.write(img)
