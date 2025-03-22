#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np

import json
import time
import h5py
import sys

from diffusion_score import compute_diffusion_score
# from plot_tsne_umap import plot_tsne, plot_scree

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
    parser.add_argument('-me', '--metric', type=str, default='diffusion', 
                        help='name of the method for measuring transferability')   
    parser.add_argument('--output-dir', type=str, default='./Diffusion_score', 
                        help='dir of output score')
    parser.add_argument('--t', type=float, default=10, help='diffusion time')
    parser.add_argument('--k', type=int, default=20, help='Connectivity for diffusion')
    parser.add_argument('--num_components', type=int, default=32, help='number of components used')
    parser.add_argument('--backbone', action='store_true', default=False, help='backbone features')

    args = parser.parse_args()   
    pprint(args)


    datasets = [  "cifar","caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397","patch_camelyon", 
                "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab","kitti", 
                "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]

    techniques = ["ADAPTER", "LORA", "convpass", "convpassattn", "facttt", "facttk","VPT", "bitfit","NOAH"]

    for metric in [f'{args.metric}']:

        for dataset in datasets:
            start_time = time.time()
            score_dict_intra = {}
            score_dict_inter = {}
            eq_dict = {}

            if args.backbone:
                fpath_dir = os.path.join(args.output_dir,'backbone', metric)
            else:
                fpath_dir = os.path.join(args.output_dir,'peft', metric)
            
            os.makedirs(fpath_dir,exist_ok=True)
        
            intra_fpath = os.path.join(fpath_dir, f'{dataset}_{args.t}_{args.k}_{args.num_components}_intra_metrics.json')
        
            inter_fpath = os.path.join(fpath_dir, f'{dataset}_{args.t}_{args.k}_{args.num_components}_inter_metrics.json')


            if not os.path.exists(intra_fpath):
                save_score(score_dict_intra, intra_fpath)
            else:
                with open(intra_fpath, "r") as f:
                    score_dict_intra = json.load(f)

            if not os.path.exists(inter_fpath):
                save_score(score_dict_inter, inter_fpath)
            else:
                with open(inter_fpath, "r") as f:
                    score_dict_inter = json.load(f)


            for technique in techniques:

                basedir = "./features"

                if args.backbone:
                    tokens_filename = f"{basedir}/{technique}_peft/{dataset}/features_backbone.h5"
                else:
                    tokens_filename = f"{basedir}/{technique}_peft/{dataset}/features_peft.h5"
                     
                print(f"Loading features from {tokens_filename}")

                with h5py.File(tokens_filename, 'r') as f:
                  

                    tokens = np.array(f['features'])  
                    labels = np.array(f['labels'])

                    # remove classes with less than 10 samples
                    unique_labels, counts = np.unique(labels, return_counts=True)
                    valid_labels = unique_labels[counts >= 10]
                    valid_indices = np.isin(labels, valid_labels)

                    tokens = tokens[valid_indices]
                    labels = labels[valid_indices]


                #     plot_tsne(tokens, labels, technique, dataset)
                #     plot_scree(tokens, labels, technique, dataset)

                    if len(labels) > 10000:
                        np.random.seed(42)
                        indices = np.random.choice(len(labels), 10000, replace=False)
                        tokens = tokens[indices]
                        labels = labels[indices]

                    labels = map_labels(labels)

                    tokens = tokens / np.linalg.norm(tokens, axis=-1, keepdims=True)


            #     # technique = standardize_technique_name(technique)

                score_dict_intra[technique],eigenvalues, eigenvectors = compute_diffusion_score(tokens, labels,eigenvalues =None, eigenvectors=None, k=args.k,t=args.t,num_components=args.num_components,intra=True)
                score_dict_inter[technique] = compute_diffusion_score(tokens, labels, eigenvalues=eigenvalues, eigenvectors=eigenvectors, k=args.k,t=args.t,num_components=args.num_components,intra=False)

                print(f'{metric} of {technique}: {score_dict_intra[technique]}\n')
                print(f'{metric} of {technique}: {score_dict_inter[technique]}\n')

            

            score_dict_inter['duration'] = time.time() - start_time
            results = sorted(score_dict_inter.items(), key=lambda i: i[1], reverse=True)
            print(f'Techniques ranking on {dataset} based on {metric}:')
            pprint(results)
            results = {a[0]: a[1] for a in results}
            save_score(results, inter_fpath)

            score_dict_intra['duration'] = time.time() - start_time
            results = sorted(score_dict_intra.items(), key=lambda i: i[1], reverse=True)
            print(f'Techniques ranking on {dataset} based on {metric}:')
            pprint(results)
            results = {a[0]: a[1] for a in results}
            save_score(results, intra_fpath)

            
            

