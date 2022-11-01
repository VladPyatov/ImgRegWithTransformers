import os
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from skimage.measure import ransac
from skimage.transform import AffineTransform

import deep_segmentation as ds
import rotation_alignment as ra
import affine_registration as ar
import deformable_registration as nr
import cost_functions as cf
import utils

from linear import SIFT
from networks import segmentation_network as sn
from networks import affine_network_attention as an
from networks import affine_network_simple as asimple
from networks import nonrigid_registration_network as nrn
from networks.feature_matching import quadtree
#from networks.LoFTR import loftr

@torch.no_grad()
def deephistreg(source, target, device, params):
    result_dict = dict()
    b_total_time = time.time()
    loss_single = cf.ncc_loss_global

    segmentation_mode = params['segmentation_mode']
    if segmentation_mode == "deep_segmentation":
        segmentation_params = params['segmentation_params']
        seg_model_path = segmentation_params['model_path']
        seg_model = sn.load_network(device, path=seg_model_path)
    initial_rotation = params['initial_rotation']
    if initial_rotation:
        initial_rotation_params = params['initial_rotation_params']
    affine_registration = params['affine_registration']
    if affine_registration:
        affine_registration_params = params['affine_registration_params']
        affine_model_path = affine_registration_params['model_path']
        affine_type = affine_registration_params['affine_type']
        if affine_type == "attention":
            affine_model = an.load_network(device, path=affine_model_path)
        elif affine_type == "simple":
            affine_model = asimple.load_network(device, path=affine_model_path)
        elif affine_type == 'quadtree':
            config = affine_registration_params['main_config_path']
            affine_model = quadtree.load_network(device=device, path=affine_model_path, main_config=config)
        elif affine_type == 'loftr':
            config = affine_registration_params['main_config_path']
            affine_model = loftr.load_network(device=device, path=affine_model_path)
        elif affine_type == 'sift':
            affine_model = SIFT(root_sift=False)

    nonrigid_registration = params['nonrigid_registration']
    if nonrigid_registration:
        nonrigid_registration_params = params['nonrigid_registration_params']
        nonrigid_model_path = nonrigid_registration_params['model_path']
        num_levels = nonrigid_registration_params['num_levels']
        nonrigid_models = list()
        for i in range(num_levels):
            current_path = nonrigid_model_path + "_level_" + str(i+1)
            nonrigid_models.append(nrn.load_network(device, path=current_path))

    displacement_field = torch.zeros(2, source.size(0), source.size(1)).to(device)
    # Tissue segmentation
    b_seg_time = time.time()

    if segmentation_mode == "deep_segmentation":
        source_mask, target_mask = ds.segmentation(source, target, seg_model, device=device)
        source[source_mask == 0] = 0
        target[target_mask == 0] = 0
    elif segmentation_mode == "manual":
        segmentation_params = params['segmentation_params']
        source_mask, target_mask = segmentation_params['source_mask'], segmentation_params['target_mask']
        source[source_mask == 0] = 0 
        target[target_mask == 0] = 0
    elif segmentation_mode is None:
        source, target = source, target
    else:
        raise ValueError("Unsupported segmentation mode.")
    e_seg_time = time.time()

    warped_source = source.clone()
    # Rotation alignment
    b_rot_time = time.time()
    if initial_rotation:
        if segmentation_mode is not None:
            if torch.sum(source_mask) >= 0.99*source.size(0)*source.size(1):
                pass
            else:
                rot_displacement_field = ra.rotation_alignment(warped_source, target, initial_rotation_params, device=device)
                displacement_field = utils.compose_displacement_field(displacement_field, rot_displacement_field, device=device)
                warped_source = utils.warp_tensor(source, displacement_field, device=device)
        else:
            rot_displacement_field, rot_ncc = ra.rotation_alignment(warped_source, target, initial_rotation_params, device=device)
            displacement_field = utils.compose_displacement_field(displacement_field, rot_displacement_field, device=device)
            warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_rot_time = time.time()
    
    # Affine registration
    b_aff_time = time.time()
    if affine_registration:
        if affine_type == 'quadtree' or affine_type == 'loftr':
            if affine_registration_params['resize']:
                new_shape = (512, 512)
                resampled_source = utils.resample_tensor(warped_source, new_shape, device=device)
                resampled_target = utils.resample_tensor(target, new_shape, device=device)
            else:
                # smart resize
                output_max_size = 1024
                new_shape = utils.calculate_new_shape_max((warped_source.size(0), warped_source.size(1)), output_max_size)
                resampled_source = utils.resample_tensor(warped_source, new_shape, device=device)
                resampled_target = utils.resample_tensor(target, new_shape, device=device)
                # pad images for the network
                if resampled_source.shape[0] != output_max_size:
                    down_padding =  output_max_size - resampled_source.shape[0]
                else:
                    down_padding = 0
                if resampled_source.shape[1] != output_max_size:
                    right_padding =  output_max_size - resampled_source.shape[1]
                else:
                    right_padding = 0
                resampled_target = F.pad(resampled_target, (0, right_padding, 0, down_padding))
            # inference
            angles = {0:0, 1:90, 3:180, 4:270}
            best_ncc = loss_single(warped_source, target, device=device)
            best_simple_ncc = 1
            best_simple_angle = 0
            found = False
            best_affine_field  = torch.zeros(2, source.size(0), source.size(1)).to(device)
            best_warped_source = warped_source.clone()

            for angle, i in angles.items():
                if i != 0:
                    rot_disp_matrix = utils.affine2theta(utils.generate_rotation_matrix(i, resampled_source.shape[1]/2, resampled_source.shape[0]/2), resampled_source.shape)[None].to(device)
                    rot_disp_field = utils.transform_to_displacement_field(resampled_source[None][None], rot_disp_matrix, device=device)
                    rot_source = utils.warp_tensor(resampled_source, rot_disp_field, device=device)
                else:
                    rot_source = resampled_source.clone()
                rot_source = F.pad(rot_source, (0, right_padding, 0, down_padding))

                simple_ncc = loss_single(rot_source, resampled_target, device=device)
                print(f'{i}-S_NCC:{simple_ncc:.5f}')
                if simple_ncc < best_simple_ncc:
                    best_simple_ncc = simple_ncc
                    best_simple_angle = i
                
                batch = {'image0': rot_source[None][None], 'image1': resampled_target[None][None]}
                affine_model(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts0[:, 0] = mkpts0[:, 0] / new_shape[1] * warped_source.shape[1]
                mkpts0[:, 1] = mkpts0[:, 1] / new_shape[0] * warped_source.shape[0]
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mkpts1[:, 0] = mkpts1[:, 0] / new_shape[1] * warped_source.shape[1]
                mkpts1[:, 1] = mkpts1[:, 1] / new_shape[0] * warped_source.shape[0]
                try:
                    (matrix, inliers) = cv2.estimateAffine2D(mkpts1, mkpts0, method=cv2.LMEDS, maxIters=10000)
                    # (model, inliers) = ransac((mkpts1, mkpts0), AffineTransform, min_samples=3, max_trials=10000,
                    #                               residual_threshold=max(warped_source.shape[:2]) * 0.01)
                    # matrix = model.params[:2, :]
                    matrix = torch.from_numpy(matrix).to(device)
                    theta = utils.affine2theta(matrix, warped_source.shape)
                    theta = theta[None].to(device)
                    # generate transform
                    affine_displacement_field = utils.transform_to_displacement_field(warped_source.view(1, 1, warped_source.size(0), warped_source.size(1)), theta, device=device)
                    if i != 0:
                        rot_disp_matrix = utils.affine2theta(utils.generate_rotation_matrix(i, warped_source.shape[1]/2, warped_source.shape[0]/2), warped_source.shape)[None].to(device)
                        rot_disp_field = utils.transform_to_displacement_field(warped_source[None][None], rot_disp_matrix, device=device)
                        final_disp_field = utils.compose_displacement_field(rot_disp_field, affine_displacement_field, device=device)
                    else:
                        final_disp_field = affine_displacement_field

                    warp = utils.warp_tensor(source, final_disp_field, device=device)
                    ncc = loss_single(warp, target, device=device)
                    print(f'{i}-NCC:{ncc:.5f}')
                    if ncc < best_ncc:
                        found = True
                        best_ncc = ncc
                        best_affine_field = final_disp_field
                        best_warped_source = warp
                except:
                    # no points were detected
                    continue
            # affine_displacement_field = utils.transform_to_displacement_field(warped_source.view(1, 1, warped_source.size(0), warped_source.size(1)), theta, device=device)
            # displacement_field = utils.compose_displacement_field(displacement_field, affine_displacement_field, device=device)
            # warped_source = utils.warp_tensor(source, displacement_field, device=device)
            if found:
                displacement_field = best_affine_field
                warped_source = best_warped_source
            else:
                rot_disp_matrix = utils.affine2theta(utils.generate_rotation_matrix(best_simple_angle, source.shape[1]/2, source.shape[0]/2), source.shape)[None].to(device)
                displacement_field = utils.transform_to_displacement_field(source[None][None], rot_disp_matrix, device=device)
                warped_source = utils.warp_tensor(source, displacement_field, device=device)
        
        elif affine_type == 'sift':
            output_max_size = 512
            new_shape = utils.calculate_new_shape_max((warped_source.size(0), warped_source.size(1)), output_max_size)
            resampled_source = utils.resample_tensor(warped_source, new_shape, device=device)
            resampled_target = utils.resample_tensor(target, new_shape, device=device)
            source_sift = utils.normalize(resampled_source.cpu().numpy()).clip(0,1) * 255
            target_sift = utils.normalize(resampled_target.cpu().numpy()).clip(0,1) * 255
            try:
                mkpts0, mkpts1 = affine_model.extract_and_match_keypoints(source_sift.astype(np.uint8), target_sift.astype(np.uint8))
                mkpts0[:, 0] = mkpts0[:, 0] / new_shape[0] * warped_source.shape[0]
                mkpts0[:, 1] = mkpts0[:, 1] / new_shape[1] * warped_source.shape[1]
                mkpts1[:, 0] = mkpts1[:, 0] / new_shape[0] * warped_source.shape[0]
                mkpts1[:, 1] = mkpts1[:, 1] / new_shape[1] * warped_source.shape[1]
                (matrix, inliers) = cv2.estimateAffine2D(mkpts1, mkpts0, method=cv2.LMEDS, maxIters=10000)
                matrix = torch.from_numpy(matrix).to(device)
                theta = utils.affine2theta(matrix, warped_source.shape)
                theta = theta[None].to(device)
                affine_displacement_field = utils.transform_to_displacement_field(warped_source.view(1, 1, warped_source.size(0), warped_source.size(1)), theta, device=device)
                displacement_field = utils.compose_displacement_field(displacement_field, affine_displacement_field, device=device)
                warped_source = utils.warp_tensor(source, displacement_field, device=device)
            except:
                print('No points were found')
        else:
            affine_displacement_field = ar.affine_registration(warped_source, target, affine_model, device=device)
            displacement_field = utils.compose_displacement_field(displacement_field, affine_displacement_field, device=device)
            warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_aff_time = time.time()
    
    # Nonrigid registration
    b_nr_time = time.time()
    if nonrigid_registration:
        nonrigid_displacement_field = nr.nonrigid_registration(warped_source, target, nonrigid_models, nonrigid_registration_params, device=device)
        displacement_field = utils.compose_displacement_field(displacement_field, nonrigid_displacement_field, device=device)
        warped_source = utils.warp_tensor(source, displacement_field, device=device)
    else:
        pass
    e_nr_time = time.time()

    e_total_time = time.time()
    result_dict['total_time'] = e_total_time - b_total_time
    result_dict['seg_time'] = e_seg_time - b_seg_time
    result_dict['rot_time'] = e_rot_time - b_rot_time
    result_dict['aff_time'] = e_aff_time - b_aff_time
    result_dict['nr_time'] = e_nr_time - b_nr_time
    return source, target, warped_source, displacement_field, result_dict

