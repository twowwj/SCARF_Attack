#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.poisoned_dataset_readers import poisonedSceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class PoisonedScene:
    """支持毒化的场景类"""

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, poison_config=None, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        初始化毒化场景
        Args:
            args: 模型参数
            gaussians: 高斯模型
            poison_config: 毒化配置
            load_iteration: 加载的迭代次数
            shuffle: 是否打乱数据
            resolution_scales: 分辨率缩放
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.poison_config = poison_config

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 根据毒化配置选择数据集读取器
        if poison_config and poison_config.poison_ratio > 0:
            print("Using poisoned dataset loader")
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = poisonedSceneLoadTypeCallbacks["Colmap"](
                    args.source_path, args.images, args.eval,
                    poison_config.poison_ratio, poison_config.trigger_path,
                    poison_config.trigger_position, poison_config.trigger_type,
                    poison_config.trigger_size, synthetic_trigger_type=poison_config.synthetic_trigger_type
                )
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = poisonedSceneLoadTypeCallbacks["Blender"](
                    args.source_path, args.white_background, args.eval,
                    poison_config.poison_ratio, poison_config.trigger_path,
                    poison_config.trigger_position, poison_config.trigger_type,
                    poison_config.trigger_size, synthetic_trigger_type=poison_config.synthetic_trigger_type
                )
            elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
                print("Found metadata.json file, assuming multi scale Blender data set!")
                scene_info = poisonedSceneLoadTypeCallbacks["Multi-scale"](
                    args.source_path, args.white_background, args.eval,
                    poison_config.poison_ratio, poison_config.trigger_path,
                    poison_config.trigger_position, poison_config.trigger_type,
                    poison_config.trigger_size, synthetic_trigger_type=poison_config.synthetic_trigger_type
                )
            else:
                assert False, "Could not recognize scene type!"
        else:
            # 使用原始数据集读取器
            print("Using clean dataset loader")
            if os.path.exists(os.path.join(args.source_path, "sparse")):
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
            elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
                print("Found metadata.json file, assuming multi scale Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
            else:
                assert False, "Could not recognize scene type!"

        # 保存毒化统计信息到数据集对象
        if hasattr(scene_info, 'poison_stats'):
            args.poison_stats = scene_info.poison_stats

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPoisonStats(self):
        """获取毒化统计信息"""
        if hasattr(self, 'poison_config') and self.poison_config:
            return getattr(self.poison_config, 'poison_stats', None)
        return None 