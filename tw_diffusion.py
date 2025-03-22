#!/usr/bin/env python
# coding: utf-8

from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric)
from scipy.stats import kendalltau
import json
import os
from datetime import datetime
import pandas as pd



def compute_kendall_tau(datasets, techniques, metric, data_dict,args):
    
    results = []
    for dataset in datasets:

        score_path = f'{args.output_dir}/{metric}/{dataset}_metrics.json'
        
        if not os.path.exists(score_path):
            
            print(f"File not found: {score_path}")
            continue
        
        with open(score_path, 'r') as file:
            metric_scores = json.load(file)
            # metric_scores,time = load_score(score_path)
            # print("time",time)
            
            if 'duration' in metric_scores.keys():
                del metric_scores['duration']

            # metric_scores = {technique_mapping[k]: v for k, v in metric_scores.items()}
            # print("metric_scores",metric_scores)

            # if 'bitfit' in metric_scores.keys():
            #     metric_scores['BitFit'] = metric_scores['bitfit']
            #     del metric_scores['bitfit']

                

        # print("metric_scores",metric_scores)
        
        accuracies = []
        scores = []
        for technique in techniques:
            accuracies.append(data_dict[dataset][technique])
            scores.append(metric_scores[technique])
            
            

        # tau, _ = kendalltau(accuracies, scores)
        w_tau = w_kendall_metric(metric_scores, data_dict, dataset)
        print(f"Weighted Kendall Tau for {dataset}: {w_tau}")
        # print(f"Kendall Tau for {dataset}: {tau}")
        results.append({
            'Dataset': dataset,
            'Weighted_Kendall_Tau': w_tau
        })
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate transferability metrics.')
    parser.add_argument('-d', '--dataset', type=str, default='deepcluster-v2', 
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-me', '--metric', type=str, default='logme', 
                        help='name of used transferability metric')
    parser.add_argument('--output-dir', type=str, default='./scores', 
                        help='dir of output score')


    args = parser.parse_args()


    
    data_dict_new = {
    "caltech101": {
        "ADAPTER": 88.78,
        "LORA": 88.82,
        "convpass": 90.91,
        "convpassattn": 91.77,
        "facttt": 91.30,
        "facttk": 91.39,
        "VPT": 89.24,
        "bitfit": 86.88,
        "NOAH": 91.87
    },
    "cifar": {
        "ADAPTER": 65.39,
        "LORA": 67.94,
        "convpass": 72.65,
        "convpassattn": 70.58,
        "facttt": 71.08,
        "facttk": 71.21,
        "VPT": 64.92,
        "bitfit": 66.23,
        "NOAH": 73.49
    },
    "dtd": {
        "ADAPTER": 69.82,
        "LORA": 69.16,
        "convpass": 74.65,
        "convpassattn": 74.15,
        "facttt": 74.11,
        "facttk": 73.00,
        "VPT": 68.78,
        "bitfit": 67.48,
        "NOAH": 73.42
    },
    "oxford_flowers102": {
        "ADAPTER": 97.62,
        "LORA": 97.54,
        "convpass": 99.17,
        "convpassattn": 98.79,
        "facttt": 98.55,
        "facttk": 98.42,
        "VPT": 99.42,
        "bitfit": 97.81,
        "NOAH": 99.57
    },
    "oxford_iiit_pet": {
        "ADAPTER": 91.11,
        "LORA": 90.34,
        "convpass": 91.94,
        "convpassattn": 92.18,
        "facttt": 91.34,
        "facttk": 91.45,
        "VPT": 90.97,
        "bitfit": 89.12,
        "NOAH": 90.80
    },
    "svhn": {
        "ADAPTER": 91.01,
        "LORA": 91.20,
        "convpass": 91.42,
        "convpassattn": 91.83,
        "facttt": 89.47,
        "facttk": 89.91,
        "VPT": 91.25,
        "bitfit": 88.50,
        "NOAH": 89.32
    },
    "sun397": {
        "ADAPTER": 53.20,
        "LORA": 53.13,
        "convpass": 52.94,
        "convpassattn": 53.43,
        "facttt": 53.28,
        "facttk": 53.20,
        "VPT": 52.12,
        "bitfit": 51.27,
        "NOAH": 52.06
    },
    "patch_camelyon": {
        "ADAPTER": 85.77,
        "LORA": 84.28,
        "convpass": 88.19,
        "convpassattn": 89.13,
        "facttt": 88.78,
        "facttk": 87.59,
        "VPT": 86.23,
        "bitfit": 85.87,
        "NOAH": 87.23
    },
    "eurosat": {
        "ADAPTER": 96.93,
        "LORA": 97.01,
        "convpass": 97.22,
        "convpassattn": 97.29,
        "facttt": 97.21,
        "facttk": 97.43,
        "VPT": 97.13,
        "bitfit": 94.82,
        "NOAH": 97.05
    },
    "resisc45": {
        "ADAPTER": 84.03,
        "LORA": 84.67,
        "convpass": 86.61,
        "convpassattn": 86.82,
        "facttt": 85.42,
        "facttk": 85.63,
        "VPT": 85.89,
        "bitfit": 82.43,
        "NOAH": 86.37
    },
    "diabetic_retinopathy": {
        "ADAPTER": 75.60,
        "LORA": 73.78,
        "convpass": 73.49,
        "convpassattn": 75.44,
        "facttt": 73.25,
        "facttk": 73.86,
        "VPT": 74.00,
        "bitfit": 73.19,
        "NOAH": 75.14
    },
    "clevr_count": {
        "ADAPTER": 82.48,
        "LORA": 82.17,
        "convpass": 83.17,
        "convpassattn": 83.09,
        "facttt": 82.96,
        "facttk": 82.2,
        "VPT": 82.91,
        "bitfit": 80.37,
        "NOAH": 82.83
    },
    "clevr_dist": {
        "ADAPTER": 65.90,
        "LORA": 64.31,
        "convpass": 65.19,
        "convpassattn": 65.48,
        "facttt": 65.61,
        "facttk": 65.94,
        "VPT": 65.48,
        "bitfit": 63.73,
        "NOAH": 66.22
    },
    "dmlab": {
        "ADAPTER": 50.97,
        "LORA": 50.62,
        "convpass": 51.93,
        "convpassattn": 51.88,
        "facttt": 51.07,
        "facttk": 52.39,
        "VPT": 50.14,
        "bitfit": 47.91,
        "NOAH": 50.26
    },
    "kitti": {
        "ADAPTER": 78.10,
        "LORA": 78.93,
        "convpass": 81.86,
        "convpassattn": 78.90,
        "facttt": 78.19,
        "facttk": 76.09,
        "VPT": 73.80,
        "bitfit": 75.11,
        "NOAH": 80.06
    },
    "dsprites_loc": {
        "ADAPTER": 82.68,
        "LORA": 82.01,
        "convpass": 85.12,
        "convpassattn": 84.29,
        "facttt": 86.35,
        "facttk": 86.19,
        "VPT": 83.41,
        "bitfit": 78.46,
        "NOAH": 85.99
    },
    "dsprites_ori": {
        "ADAPTER": 54.48,
        "LORA": 54.30,
        "convpass": 54.13,
        "convpassattn": 54.72,
        "facttt": 54.26,
        "facttk": 54.18,
        "VPT": 52.63,
        "bitfit": 51.58,
        "NOAH": 53.84
    },
    "smallnorb_azi": {
        "ADAPTER": 35.16,
        "LORA": 37.18,
        "convpass": 38.49,
        "convpassattn": 38.48,
        "facttt": 37.84,
        "facttk": 38.30,
        "VPT": 34.33,
        "bitfit": 22.20,
        "NOAH": 34.91
    },
    "smallnorb_ele": {
        "ADAPTER": 43.16,
        "LORA": 43.08,
        "convpass": 45.79,
        "convpassattn": 43.18,
        "facttt": 43.55,
        "facttk": 43.08,
        "VPT": 42.88,
        "bitfit": 37.47,
        "NOAH": 44.81
    }
}

    # technique_mapping = {
    #     "bitfit": "BitFit",
    #     "ADAPTER": "Adapter",
    #     "LORA": "LoRA",
    #     "NOAH": "NOAH",
    #     "VPT": "VPT-Deep",
    #     "convpass": "Convpass",
    #     "convpassattn": "Convpass_attn",
    #     "facttt": "FacT-TT<16",
    #     "facttk": "FacT-TK32"
    # }


    datasets = ["caltech101", "cifar", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397", "patch_camelyon", "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab", "kitti", "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]
    techniques = ["ADAPTER", "LORA",  "convpass", "convpassattn", "facttt", "facttk","VPT","bitfit","NOAH"]
    metric = args.metric
    
    kendall_results = compute_kendall_tau(
        datasets, techniques, metric, data_dict_new, args
    )


