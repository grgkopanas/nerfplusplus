import numpy as np
import json
import random
from os import makedirs, path
from errno import EEXIST
import os
import sys
import shutil

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def make_nerf_folder(cams_dict, folder, in_rgbs=None):
    intrinsic_folder = os.path.join(folder, "intrinsics")
    mkdir_p(intrinsic_folder)
    pose_folder = os.path.join(folder, "pose")
    mkdir_p(pose_folder)
    if in_rgbs:
        rgb_folder = os.path.join(folder, "rgb")
        mkdir_p(rgb_folder)
    for idx, packed in enumerate(cams_dict.items()):
        cam_name, cam = packed
        with open(os.path.join(intrinsic_folder, format(idx, '05d')+ ".txt"), 'w') as f:
            for k in cam["K"]:
                f.write(str(k) + " ")

        W2C = np.array(cam["W2C"]).reshape((4,4))
        C2W = np.linalg.inv(W2C)
        with open(os.path.join(pose_folder, format(idx, '05d') + ".txt"), 'w') as f:
            for k in C2W.flatten():
                f.write(str(k) + " ")
        if in_rgbs:
            shutil.copyfile(os.path.join(in_rgbs, cam_name),
                            os.path.join(rgb_folder, format(idx, '05d') + "." + cam_name.split(".")[-1]))

def convert_colmaprunner_to_nerf(input_folder, output_folder, val, test):
    json_cameras = os.path.join(input_folder, "kai_cameras_normalized.json")
    images_folder = os.path.join(input_folder, "images")

    fp = open(json_cameras)
    in_cam_dict = json.load(fp)
    fp.close()

    in_cam_dict_keys = list(in_cam_dict.keys())
    random.shuffle(in_cam_dict_keys)

    test_dict = {cam_name: in_cam_dict[cam_name] for cam_name in in_cam_dict_keys[:test]}
    valid_dict = {cam_name: in_cam_dict[cam_name] for cam_name in in_cam_dict_keys[test:test+val]}
    train_dict = {cam_name: in_cam_dict[cam_name] for cam_name in in_cam_dict_keys[test+val:]}

    nerfscene_folder = os.path.join(output_folder, "test")
    mkdir_p(nerfscene_folder)
    make_nerf_folder(test_dict, nerfscene_folder, images_folder)

    nerfscene_folder = os.path.join(output_folder, "validation")
    mkdir_p(nerfscene_folder)
    make_nerf_folder(valid_dict, nerfscene_folder, images_folder)

    nerfscene_folder = os.path.join(output_folder, "train")
    mkdir_p(nerfscene_folder)
    make_nerf_folder(train_dict, nerfscene_folder, images_folder)


if __name__ == "__main__":
    scene_folder = "F:/gkopanas/scenes/deep_blending/museum/Museum-1_nerf++/"
    convert_colmaprunner_to_nerf(scene_folder)