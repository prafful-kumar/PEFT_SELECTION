#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint
import h5py
import torch
import models.group1 as models
import numpy as np
import json
import time

from metrics import NLEEP, LogME_Score, SFDA_Score,LDA_Score,NCTI_Score,get_gbc_score

def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)


def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

def map_labels(y):
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    mapped_y = np.array([label_map[label] for label in y])
    return mapped_y

# # Add this function to standardize technique names
# def standardize_technique_name(technique):
#     mapping = {
#         "ADAPTER": "Adapter",
#         "LORA": "LoRA",
#         "VPT": "VPT-Deep",
#         "convpass": "Convpass",
#         "convpassattn": "Convpass_attn",
#         "facttt": "FacT-TT<16",
#         "facttk": "FacT-TK32",
#         "NOAH": "NOAH",
#         "bitfit": "BitFit",
#     }
#     return mapping.get(technique, technique)  # Default to original if not found


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transferability score.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme', 
                        help='name of the method for measuring transferability')   
    parser.add_argument('--nleep-ratio', type=float, default=5, 
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--output-dir', type=str, default='./TE_previous', 
                        help='dir of output score')


    args = parser.parse_args()   
    pprint(args)


    datasets = [  "cifar","caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397","patch_camelyon", 
                "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab","kitti", 
                "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]
    techniques = ["ADAPTER", "LORA", "convpass", "convpassattn", "facttt", "facttk","VPT", "bitfit","NOAH"]

    for metric in [f'{args.metric}']:

        for dataset in datasets:
            start_time = time.time()
            score_dict = {}
            eq_dict = {}
            fpath = os.path.join(args.output_dir, metric)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            fpath = os.path.join(fpath, f'{dataset}_metrics.json')

            if not os.path.exists(fpath):
                save_score(score_dict, fpath)
            else:
                with open(fpath, "r") as f:
                    score_dict = json.load(f)


            for technique in techniques:


                basedir = "../../features"
                tokens_filename = f"{basedir}/{technique}_peft/{dataset}/features_peft.h5"

                print(f"Loading features from {tokens_filename}")

                with h5py.File(tokens_filename, 'r') as f:
                  

                    tokens = np.array(f['features'])  
                    labels = np.array(f['labels'])

                    labels = map_labels(labels)

                # technique = standardize_technique_name(technique)

                if metric == 'logme':
                    score_dict[technique] = LogME_Score(tokens, labels)
                elif metric == 'nleep':
                    score_dict[technique] = NLEEP(tokens, labels, args.nleep_ratio)
                elif metric == 'sfda':
                    score_dict[technique] = SFDA_Score(tokens, labels)

                elif metric == 'lda':
                    score_dict[technique] = LDA_Score(tokens, labels)

                elif metric == 'ncti':
                    score_dict[technique] = NCTI_Score(tokens, labels)

                elif metric == 'gbc':
                    score_dict[technique] = get_gbc_score(torch.from_numpy(tokens), torch.from_numpy(labels), 'diagonal')


                print(f'{metric} of {technique}: {score_dict[technique]}\n')

                # save_score(score_dict, fpath)

            score_dict['duration'] = time.time() - start_time

            # results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
            # print(f'techniques ranking on {dataset} based on {metric}: ')
            # pprint(results)
            # results = {a[0]: a[1] for a in results}
            # save_score(results, fpath)


            if metric in ['ncti']:
                
                all_score = []
                cls_score = []
                cls_compact = []

                for model in techniques:
                    # model = standardize_technique_name(model)
                    all_score.append(score_dict[model][0])
                    cls_score.append(score_dict[model][1])
                    cls_compact.append(score_dict[model][2])
                    
                all_score = np.array(all_score)
                cls_score = np.array(cls_score)
                cls_compact = np.array(cls_compact)

                all_score_min = all_score.min()
                all_score_div = all_score.max() - all_score.min()

                cls_score_min = cls_score.min()
                cls_score_div = cls_score.max() - cls_score.min()

                cls_compact_min = cls_compact.min()
                cls_compact_div = cls_compact.max() - cls_compact_min

                for model in techniques:
                    # model = standardize_technique_name(model)
                    print(model)
                    mascore = (score_dict[model][0] - all_score_min)/all_score_div # Seli score
                    mcscore = (score_dict[model][1] - cls_score_min)/cls_score_div # NCC score
                    cpscore = (score_dict[model][2] - cls_compact_min)/cls_compact_div # class compactness score
                    print(f"ma score for {model}: {mascore}")
                    print(f"mc score for {model}: {mcscore}")
                    print(f"cp score for {model}: {cpscore}")

                    score_dict[model] = mcscore  + mascore - cpscore

            save_score(score_dict, fpath)
                
            results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
            print(f'Models ranking on {dataset} based on {metric}: ')
            pprint(results)
            results = {a[0]: a[1] for a in results}
            save_score(results, fpath)