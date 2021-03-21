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

def convert_colmaprunner_to_nerf(scene_folder):
    json_cameras = os.path.join(scene_folder, "posed_images", "kai_cameras_normalized.json")
    output_folder = os.path.join(scene_folder, "nerfscene")
    images_folder = os.path.join(scene_folder, "posed_images", "images")

    fp = open(json_cameras)
    in_cam_dict = json.load(fp)
    fp.close()

    in_cam_dict_keys = list(in_cam_dict.keys())
    random.shuffle(in_cam_dict_keys)

    test_keys = in_cam_dict_keys[:2]
    valid_keys = in_cam_dict_keys[2:4]
    train_keys = in_cam_dict_keys[4:]

    for idx, cam_name in enumerate(in_cam_dict_keys):
        folder = None
        if cam_name in test_keys:
            folder = "test"
        elif cam_name in valid_keys:
            folder = "validation"
        elif cam_name in train_keys:
            folder = "train"
        else:
            print("Camera {} not found".format(cam_name))
            exit(-1)
        nerfscene_folder = os.path.join(output_folder, folder)
        intrinsic_folder = os.path.join(nerfscene_folder, "intrinsics")
        pose_folder = os.path.join(nerfscene_folder, "pose")
        rgb_folder = os.path.join(nerfscene_folder, "rgb")
        mkdir_p(nerfscene_folder)
        mkdir_p(intrinsic_folder)
        mkdir_p(pose_folder)
        mkdir_p(rgb_folder)
        with open(os.path.join(intrinsic_folder, format(idx, '05d')+ ".txt"), 'w') as f:
            for k in in_cam_dict[cam_name]["K"]:
                f.write(str(k) + " ")

        W2C = np.array(in_cam_dict[cam_name]["W2C"]).reshape((4,4))
        C2W = np.linalg.inv(W2C)
        with open(os.path.join(pose_folder, format(idx, '05d') + ".txt"), 'w') as f:
            for k in C2W.flatten():
                f.write(str(k) + " ")

        shutil.copyfile(os.path.join(images_folder, cam_name),
                        os.path.join(rgb_folder, format(idx, '05d') + "." + cam_name.split(".")[-1]))

if __name__ == "__main__":
    scene_folder = "F:/gkopanas/scenes/deep_blending/museum/Museum-1_nerf++/"
    convert_colmaprunner_to_nerf(scene_folder)