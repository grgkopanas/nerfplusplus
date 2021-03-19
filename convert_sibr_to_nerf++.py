from colmap_runner.extract_sfm import extract_all_to_dir
import os
from os import makedirs, path
from errno import EEXIST
import os
import sys
import shutil
from normalize_cam_dict import normalize_cam_dict
from convert_colmaprunner_to_nerf import convert_colmaprunner_to_nerf


mvs_dir = "F:/gkopanas/scenes/deep_blending/museum/Museum-1_perview/colmap/stereo"
out_dir = "F:/gkopanas/scenes/deep_blending/museum/Museum-1_nerf++"

makedirs(os.path.join(out_dir, 'posed_images'), exist_ok=True)

extract_all_to_dir(os.path.join(mvs_dir, 'sparse'), os.path.join(out_dir, 'posed_images'), ext=".txt")
undistorted_img_dir = os.path.join(mvs_dir, 'images')
posed_img_dir_link = os.path.join(out_dir, 'posed_images/images')
if os.path.exists(posed_img_dir_link):
    shutil.rmtree(posed_img_dir_link)
shutil.copytree(undistorted_img_dir, posed_img_dir_link)
# normalize average camera center to origin, and put all cameras inside the unit sphere
normalize_cam_dict(os.path.join(out_dir, 'posed_images/kai_cameras.json'),
                   os.path.join(out_dir, 'posed_images/kai_cameras_normalized.json'))

convert_colmaprunner_to_nerf(out_dir)
