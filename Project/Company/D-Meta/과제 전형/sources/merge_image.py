GLOBAL_SEED = 42

import os
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)

import sys
import numpy as np
from numpy import random as np_rnd
import pandas as pd
import random as rnd
from itertools import  permutations
from datetime import datetime

import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as tvf
from skimage.feature import hog, local_binary_pattern
import cv2

import optuna
from optuna import Trial, create_study
from optuna.samplers import NSGAIISampler
import operator

import warnings
warnings.filterwarnings('ignore')

from helper_functions import *


class CFG:
    debug = False
    image_root_path = "./data/"
    # h, w
    target_grid = (None, None)
    operations = {
        "mirroring": lambda x: tvf.hflip(x),
        "flipping": lambda x: tvf.vflip(x),
        "rotation": lambda x: tvf.rotate(x, 90, expand=True)[:, :x.shape[2], :x.shape[1]],
    }
    inverse_operations = {
        "mirroring": lambda x: tvf.hflip(x),
        "flipping": lambda x: tvf.vflip(x),
        "rotation": lambda x: tvf.rotate(x, -90, expand=True)[:, :x.shape[2], :x.shape[1]],
    }
    shuffled_image_name_length = 4
    max_loss_value = 1000.0
    hog_params = {
        "orientations": 9,
        "cells_per_block": (3, 3),
        "pixels_per_cell": (3, 3),
        "channel_axis": 0,
    }
    lbp_params = {
        "P": 3 * 3,
        "R": 3,
        "method": "uniform",
    }


def get_pixel_histogram_feature_vector(x, n_bins=64):
    output = []
    for i in x:
        histogram, _ = torch.histogram(i, bins=n_bins, range=(0, 1))
        output.append(histogram / histogram.sum())
    return torch.cat(output, dim=-1)


def get_hog_feature_vector(x):
    return torch.from_numpy(hog(x.detach().cpu().numpy(), **CFG.hog_params).flatten())


def get_lbp_feature_vector(x):
    return torch.from_numpy(local_binary_pattern(tvf.rgb_to_grayscale(x).squeeze(dim=0).detach().cpu().numpy(), **CFG.lbp_params).flatten())


def get_pixel_variety(x1, x2, n_bins=64):
    pixel_variety = torch.stack([torch.histogram(i, bins=n_bins, range=(0, 1))[0] for i in x1]) + torch.stack([torch.histogram(i, bins=n_bins, range=(0, 1))[0] for i in x2])
    return 1 / (pixel_variety.flatten().std().item() + 1.0)


def get_local_blended_score(x1, x2):
    score = [
        torch.sqrt(torch.pow(x1.flatten() - x2.flatten(), 2).mean()).item() / 2.0,
        torch.sqrt(torch.pow(get_pixel_histogram_feature_vector(x1) - get_pixel_histogram_feature_vector(x2), 2).mean()).item() * 20.0,
        torch.sqrt(torch.pow(get_hog_feature_vector(x1) - get_hog_feature_vector(x2), 2).mean()).item(),
        torch.sqrt(torch.pow(get_lbp_feature_vector(x1) - get_lbp_feature_vector(x2), 2).mean()).item() / 100.0,
        get_pixel_variety(x1, x2),
    ]
    return np.mean(score)


def get_consistency_score(x, order):
    if isinstance(x, list):
        x = torch.stack(x, dim=0)

    rows_global = []
    rows_local_binary = []
    rows_local_lbp = []
    row_idx = np.array([int(np.floor(order[i] / (CFG.target_grid[1]))) for i in range(len(x))])
    for i in range(CFG.target_grid[0]):
        row_image = x[row_idx == i]
        row_order = order[row_idx == i]

        rows_global.append(torch.cat([j for j in row_image[np.argsort(row_order)]], dim=-1))
        rows_local_binary.append(torch.cat([torch.from_numpy(cv2.threshold(np.array(transforms.ToPILImage()(tvf.rgb_to_grayscale(j).squeeze(dim=0))), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]) for j in row_image[np.argsort(row_order)]], dim=-1))
        rows_local_lbp.append(torch.cat([torch.from_numpy(local_binary_pattern(tvf.rgb_to_grayscale(j).squeeze(dim=0).detach().cpu().numpy(), **CFG.lbp_params).flatten()) for j in row_image[np.argsort(row_order)]], dim=-1))

    rows_global = torch.cat(rows_global, dim=1)
    rows_global_binary = torch.from_numpy(cv2.threshold(np.array(transforms.ToPILImage()(tvf.rgb_to_grayscale(rows_global).squeeze(dim=0))), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]).to(torch.float32)
    rows_global_lbp = torch.from_numpy(local_binary_pattern(tvf.rgb_to_grayscale(rows_global).squeeze(dim=0).detach().cpu().numpy(), **CFG.lbp_params).flatten()).to(torch.float32)
    rows_local_binary = torch.cat(rows_local_binary, dim=0).to(torch.float32)
    rows_local_lbp = torch.cat(rows_local_lbp, dim=0).to(torch.float32)

    output = [
        {"thresholding_consistency": torch.sqrt(torch.pow(rows_global_binary - rows_local_binary, 2).mean()).item() / 200.0},
        {"lbp_consistency": torch.sqrt(torch.pow(rows_global_lbp - rows_local_lbp, 2).mean()).item() / 25.0},
    ]
    return output


def get_edge_similarity_score(transformed_image, position_order, ratio=0.05, min_edge_pixels=9):
    score = []
    for pos in position_order:

        # check height upper
        if (pos + (CFG.target_grid[1] * -1)) >= 0:
            target_image = transformed_image[position_order.index(pos + (CFG.target_grid[1] * -1))]
            pos_image = transformed_image[position_order.index(pos)]
            # if shape is not matched, return large loss value
            if target_image.shape[2] != pos_image.shape[2]:
                score.append({tuple(sorted([pos, (pos + (CFG.target_grid[1] * -1))])): CFG.max_loss_value})
            else:
                # for target image
                target_image = target_image[:, (-1 * max(min_edge_pixels, int(target_image.shape[1] * ratio))):, :]
                # for present image
                pos_image = pos_image[:, :(1 * max(min_edge_pixels, int(pos_image.shape[1] * ratio))):, :]
                target_image = tvf.vflip(target_image)
                # get l2 distance from feature vector
                target_image = tvf.vflip(target_image)
                score.append({tuple(sorted([pos, (pos + (CFG.target_grid[1] * -1))])): get_local_blended_score(target_image, pos_image)})

        # check height lower
        if (pos + (CFG.target_grid[1] * 1)) <= (CFG.target_grid[0] * CFG.target_grid[1] - 1):
            target_image = transformed_image[position_order.index(pos + (CFG.target_grid[1] * 1))]
            pos_image = transformed_image[position_order.index(pos)]
            # if shape is not matched, return large loss value
            if target_image.shape[2] != pos_image.shape[2]:
                score.append({tuple(sorted([pos, (pos + (CFG.target_grid[1] * 1))])): CFG.max_loss_value})
            else:                
                # for target image
                target_image = target_image[:, :(1 * max(min_edge_pixels, int(target_image.shape[1] * ratio))), :]
                target_image = tvf.vflip(target_image)
                # for present image
                pos_image = pos_image[:, (-1 * max(min_edge_pixels, int(pos_image.shape[1] * ratio))):, :]
                # get l2 distance from feature vector
                score.append({tuple(sorted([pos, (pos + (CFG.target_grid[1] * 1))])): get_local_blended_score(target_image, pos_image)})
            
        # check width left
        if int(np.floor((pos - 1) / CFG.target_grid[1])) == int(np.floor(pos / CFG.target_grid[1])):
            # print(44)
            target_image = transformed_image[position_order.index(pos - 1)]
            pos_image = transformed_image[position_order.index(pos)]
            # if shape is not matched, return large loss value
            if target_image.shape[1] != pos_image.shape[1]:
                # print(33)
                score.append({tuple(sorted([pos, (pos - 1)])): CFG.max_loss_value})
                # print(score)
            else:                
                # for target image
                target_image = target_image[:, :, (-1 * max(min_edge_pixels, int(target_image.shape[2] * ratio))):]
                target_image = tvf.hflip(target_image)
                # for present image
                pos_image = pos_image[:, :, :(1 * max(min_edge_pixels, int(pos_image.shape[2] * ratio)))]
                # get l2 distance from feature vector
                score.append({tuple(sorted([pos, (pos - 1)])): get_local_blended_score(target_image, pos_image)})

        # check width right
        if int(np.floor((pos + 1) / CFG.target_grid[1])) == int(np.floor(pos / CFG.target_grid[1])):
            target_image = transformed_image[position_order.index(pos + 1)]
            pos_image = transformed_image[position_order.index(pos)]
            # if shape is not matched, return large loss value
            if target_image.shape[1] != pos_image.shape[1]:
                score.append({tuple(sorted([pos, (pos + 1)])): CFG.max_loss_value})
            else:                
                # for target image
                target_image = target_image[:, :, :(1 * max(min_edge_pixels, int(target_image.shape[2] * ratio)))]
                target_image = tvf.hflip(target_image)
                # for present image
                pos_image = pos_image[:, :, (-1 * max(min_edge_pixels, int(pos_image.shape[2] * ratio))):]
                # get l2 distance from feature vector
                score.append({tuple(sorted([pos, (pos + 1)])): get_local_blended_score(target_image, pos_image)})

    score.extend(get_consistency_score(transformed_image, np.array(position_order)))
    df_score = pd.Series([list(i.values())[0] for i in score], index=[list(i.keys())[0] for i in score])
    df_score.index.name = "pair"
    df_score = df_score.groupby("pair").mean()
    return df_score.mean()


# optuna function
def optuna_objective_function(trial: Trial, output_container, image_container, resolution_range, inverse_operation_order):

    transformed_image = [i.clone() for i in image_container]
    
    # set searching parameters
    tuner_params = {}
    for i in range(len(transformed_image)):
        tuner_params[f"img{i}_opSeq"] = trial.suggest_categorical(f"img{i}_opSeq", list(range(len(inverse_operation_order))))
        for op in CFG.inverse_operations.keys():
            if op != "rotation":
                tuner_params[f"img{i}_{op}"] = trial.suggest_categorical(f"img{i}_{op}", [0, 1])
    # set resolution indicating paramter for reducing complexity
    tuner_params["resolution_h"] = trial.suggest_categorical(f"resolution_h", resolution_range)

    # transform images with seaching params & assign order
    # first, align all images with selected resolution
    for i in range(len(transformed_image)):
        for op in inverse_operation_order[tuner_params[f"img{i}_opSeq"]]:
            if op != "rotation":
                if tuner_params[f"img{i}_{op}"] == 1:
                    transformed_image[i] = CFG.inverse_operations[op](transformed_image[i])
            else:
                if transformed_image[i].shape[1] != tuner_params["resolution_h"]:
                    transformed_image[i] = CFG.inverse_operations["rotation"](transformed_image[i])

    # assign postion on images
    position_order = list(range(len(transformed_image)))

    # calcuate edge similarity score
    optuna_score = get_edge_similarity_score(transformed_image, position_order)
    
    if optuna_score < output_container["optuna_score"]:
        output_container["optuna_score"] = optuna_score
        print(f"Found best at {trial.number} trial (Score -> {optuna_score})")

    return optuna_score


class Optuna_EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


def do_optimization(seed, input_container={}):
    generic_sampler_params = {
        "population_size": 50,
        "mutation_prob": 0.05,
        "crossover_prob": 0.8,
        "swapping_prob": 0.5,
        "seed": seed,
    }

    seed_everything(seed)
    output_container = {"optuna_score": np.inf}
    optuna_direction = "minimize"
    optuna_trials = 10_000
    optuna_earlyStopping = Optuna_EarlyStoppingCallback(max(1, int(optuna_trials * 0.2)), direction=optuna_direction)
    optuna_timout = 6 * 3600
    optuna_study = create_study(direction=optuna_direction, sampler=NSGAIISampler(**generic_sampler_params))

    optuna_study.optimize(
        lambda trial: optuna_objective_function(
            trial, output_container, **input_container
        ),
        n_jobs=-1, n_trials=optuna_trials, timeout=optuna_timout, callbacks=[optuna_earlyStopping]
    )
    
    return optuna_study.best_params


def inference(output_file_name, best_params, image_container, resolution_range, inverse_operation_order):
    transformed_image = [i.clone() for i in image_container]

    for i in range(len(transformed_image)):
        for op in inverse_operation_order[best_params[f"img{i}_opSeq"]]:
            if op != "rotation":
                if best_params[f"img{i}_{op}"] == 1:
                    transformed_image[i] = CFG.inverse_operations[op](transformed_image[i])
            else:
                if transformed_image[i].shape[1] != best_params["resolution_h"]:
                    transformed_image[i] = CFG.inverse_operations["rotation"](transformed_image[i])

    position_order = np.array(range(len(transformed_image)))
    transformed_image = torch.stack(transformed_image, dim=0)

    rows = []
    row_idx = np.array([int(np.floor(position_order[i] / (CFG.target_grid[1]))) for i in range(len(transformed_image))])
    for i in range(CFG.target_grid[0]):
        row_image = transformed_image[row_idx == i]
        row_order = position_order[row_idx == i]
        rows.append(torch.cat([j for j in row_image[np.argsort(row_order)]], dim=-1))
    final_image = torch.cat(rows, dim=1)

    final_image = transforms.ToPILImage()(final_image)
    final_image.save(output_file_name)


def main():
    # parsing arguments
    target_grid_h, target_grid_w = sys.argv[1:][1], sys.argv[1:][2]
    if (not target_grid_h.isdigit()) or (not target_grid_w.isdigit()):
        print("value error on the number of grid")
        return -1
    CFG.target_grid = (int(target_grid_h), int(target_grid_w))
    input_foler_path = CFG.image_root_path + "random_" + sys.argv[1:][0].split(".")[0] + "/"
    output_file_name = sys.argv[1:][3]
    seed = int(sys.argv[1:][4]) if sys.argv[1:][4].isdigit() else int(datetime.now().timestamp())
    
    # loading sub-images
    image_container = [transforms.ToTensor()(Image.open(input_foler_path + i)) for i in sorted(os.listdir(input_foler_path))]
    # get resolution value range (there are only 2 values)
    resolution_metainfo = []
    for idx, value in enumerate(image_container):
        resolution_metainfo.append({
            "idx": idx,
            "h": value.shape[1],
            "w": value.shape[2],
        })
    resolution_metainfo = pd.DataFrame(resolution_metainfo)
    resolution_range = tuple(set(resolution_metainfo["h"].to_list() + resolution_metainfo["w"].to_list()))
    assert len(resolution_range) <= 2
    # get inverse order
    inverse_operation_order = list(permutations(CFG.inverse_operations.keys()))

    # optimization
    input_container = {
        "image_container": image_container,
        "resolution_range": resolution_range,
        "inverse_operation_order": inverse_operation_order,
    }
    best_params = do_optimization(seed, input_container=input_container)

    # inference (merging)
    inference(output_file_name, best_params, **input_container)


if __name__ == "__main__":
    main()
