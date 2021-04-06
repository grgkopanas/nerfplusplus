from colmap_runner.extract_sfm import extract_all_to_dir
import os
from os import makedirs, path
from errno import EEXIST
import os
import sys
from normalize_cam_dict import normalize_cam_dict
from convert_colmaprunner_to_nerf import convert_colmaprunner_to_nerf, make_nerf_folder
import re
import numpy as np
import math
from colmap_runner.normalize_cam_dict import transform_pose
import json
from pyquaternion import Quaternion

def readLookAtSIBR(file):
    with open(file, "r") as f:
        content = f.readlines()
    cameras = []
    for l in content:
        cam = {}
        split_line = re.split(',| |\n|=', l)
        cam["eye"] = np.array([float(n) for n in split_line[3:6]]).reshape(3)
        cam["at"] = np.array([float(n) for n in split_line[8:11]]).reshape(3)
        cam["up"] = np.array([float(n) for n in split_line[13:16]]).reshape(3)
        cam["fovY"] = float(split_line[18])
        cameras.append(cam)
    return cameras

def convert_lookAt_to_KRt(cam_lookat, w=900, h=600):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        return v / norm

    zAxis = normalize(cam_lookat["eye"] - cam_lookat["at"])
    xAxis = normalize(np.cross(cam_lookat["up"], zAxis))
    yAxis = np.cross(zAxis, xAxis)

    zAxis = -zAxis
    yAxis = -yAxis

    t = np.array([-np.dot(cam_lookat["eye"], xAxis),
                  -np.dot(cam_lookat["eye"], yAxis),
                  -np.dot(cam_lookat["eye"], zAxis)])

    R = np.empty((3,3))
    R[0,:] = xAxis
    R[1,:] = yAxis
    R[2,:] = zAxis

    #print("x:{} y:{} z:{}".format(xAxis, yAxis, zAxis))

    W2C = np.eye(4)
    W2C[:3,:3] = R
    W2C[:3, 3] = t

    sibr_focal_y = 0.5*h/math.tan(cam_lookat["fovY"] / 2.0)
    sibr_focal_x = 2*sibr_focal_y/w

    K=np.eye(4)
    K[0][0] = sibr_focal_y
    K[1][1] = sibr_focal_y
    K[0][2] = w/2.0
    K[1][2] = h/2.0

    return {"K": K, "W2C": W2C, "img_size": [w, h]}

mvs_dir = "F:/gkopanas/scenes/deep_blending/street/Street-10_perview/colmap/stereo"
out_dir = "F:/gkopanas/scenes/deep_blending/street/Street-10_nerf++_full"
path_file = "F:/gkopanas/scenes/deep_blending/street/2021street.lookat"


extract_all_to_dir(mvs_dir, os.path.join(out_dir, 'posed_images'), ext=".txt")


# normalize average camera center to origin, and put all cameras inside the unit sphere
normalize_cam_dict(os.path.join(out_dir, 'posed_images/kai_cameras.json'),
                   os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json'))

json_folder = os.path.join(out_dir, "posed_images")
out_folder = os.path.join(out_dir, "nerfscene")
convert_colmaprunner_to_nerf(input_folder=json_folder, output_folder=out_folder, val=0, test=0)

transform_dict = json.load(open(os.path.join(out_dir, "posed_images/norm_transform.json")))

path_cameras = readLookAtSIBR(path_file)
KW2C_cams = [convert_lookAt_to_KRt(cam) for cam in path_cameras]

W2CK_normalized_cams = {}
for idx, cam in enumerate(KW2C_cams):
    W2CK_normalized_cams[str(idx)] = {"K": list(cam["K"].flatten()),
                                      "W2C": list(transform_pose(cam["W2C"],
                                                                 transform_dict["translate"],
                                                                 transform_dict["scale"]).flatten()),
                                      "img_size": cam["img_size"]
                                      }
make_nerf_folder(W2CK_normalized_cams, os.path.join(out_folder, "camera_path"))
asd =123
