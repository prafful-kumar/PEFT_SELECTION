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

    parser.add_argument('--task', type=str, default='vtab', choices=['vtab', 'fgvc'],
                        help='Task type: vtab or fgvc')



    args = parser.parse_args()


    
    

    

    if args.task == 'vtab':
        datasets = ["cifar100","caltech101", "dtd", "oxford_flowers102", "oxford_iiit_pet", "svhn", "sun397","patch_camelyon", 
                    "eurosat", "resisc45", "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab","kitti", 
                    "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"]

        data_dict = {
                    "caltech101": {
                        "adapter": 88.78,
                        "lora": 88.82,
                        "convpass": 90.91,
                        "convpass_attn": 91.77,
                        "fact_tt": 91.30,
                        "fact_tk": 91.39,
                        "vptshallow": 92.24,
                        "bitfit": 86.88,
                        "noah": 91.01
                    },
                    "cifar100": {
                        "adapter": 72.39,
                        "lora": 69.94,
                        "convpass": 72.65,
                        "convpass_attn": 68.58,
                        "fact_tt": 72.08,
                        "fact_tk": 72.21,
                        "vptshallow": 73.92,
                        "bitfit": 72.83,
                        "noah": 72.49
                    },
                    "dtd": {
                        "adapter": 69.82,
                        "lora": 69.16,
                        "convpass": 73.65,
                        "convpass_attn": 73.15,
                        "fact_tt": 73.11,
                        "fact_tk": 72.00,
                        "vptshallow": 73.78,
                        "bitfit": 67.48,
                        "noah": 73.42
                    },
                    "oxford_flowers102": {
                        "adapter": 97.62,
                        "lora": 97.54,
                        "convpass": 98.17,
                        "convpass_attn": 98.79,
                        "fact_tt": 99.55,
                        "fact_tk": 98.42,
                        "vptshallow": 99.42,
                        "bitfit": 97.81,
                        "noah": 97.97
                    },
                    "oxford_iiit_pet": {
                        "adapter": 91.11,
                        "lora": 90.34,
                        "convpass": 90.94,
                        "convpass_attn": 91.18,
                        "fact_tt": 91.34,
                        "fact_tk": 91.45,
                        "vptshallow": 91.97,
                        "bitfit": 89.12,
                        "noah": 90.80
                    },
                    "svhn": {
                        "adapter": 91.01,
                        "lora": 90.67,
                        "convpass": 90.42,
                        "convpass_attn": 92.83,
                        "fact_tt": 89.47,
                        "fact_tk": 89.91,
                        "vptshallow": 91.25,
                        "bitfit": 92.50,
                        "noah": 89.32
                    },
                    "sun397": {
                        "adapter": 53.20,
                        "lora": 53.13,
                        "convpass": 52.94,
                        "convpass_attn": 53.43,
                        "fact_tt": 53.28,
                        "fact_tk": 53.20,
                        "vptshallow": 54.12,
                        "bitfit": 51.27,
                        "noah": 52.06
                    },
                    "patch_camelyon": {
                        "adapter": 85.77,
                        "lora": 88.98,
                        "convpass": 88.39,
                        "convpass_attn": 88.13,
                        "fact_tt": 87.78,
                        "fact_tk": 88.59,
                        "vptshallow": 88.23,
                        "bitfit": 85.87,
                        "noah": 87.23
                    },
                    "eurosat": {
                        "adapter": 96.93,
                        "lora": 97.91,
                        "convpass": 97.22,
                        "convpass_attn": 98.29,
                        "fact_tt": 97.21,
                        "fact_tk": 97.43,
                        "vptshallow": 97.13,
                        "bitfit": 97.82,
                        "noah": 97.05
                    },
                    "resisc45": {
                        "adapter": 84.03,
                        "lora": 84.67,
                        "convpass": 86.61,
                        "convpass_attn": 85.82,
                        "fact_tt": 85.42,
                        "fact_tk": 85.63,
                        "vptshallow": 85.89,
                        "bitfit": 82.43,
                        "noah": 85.37
                    },
                    "diabetic_retinopathy": {
                        "adapter": 74.60,
                        "lora": 73.78,
                        "convpass": 75.49,
                        "convpass_attn": 74.44,
                        "fact_tt": 73.25,
                        "fact_tk": 73.86,
                        "vptshallow": 74.00,
                        "bitfit": 74.19,
                        "noah": 73.14
                    },
                    "clevr_count": {
                        "adapter": 82.48,
                        "lora": 82.17,
                        "convpass": 83.17,
                        "convpass_attn": 82.09,
                        "fact_tt": 82.96,
                        "fact_tk": 82.2,
                        "vptshallow": 82.91,
                        "bitfit": 80.37,
                        "noah": 82.83
                    },
                    "clevr_dist": {
                        "adapter": 63.90,
                        "lora": 64.31,
                        "convpass": 66.19,
                        "convpass_attn": 65.48,
                        "fact_tt": 65.61,
                        "fact_tk": 65.94,
                        "vptshallow": 66.08,
                        "bitfit": 63.73,
                        "noah": 63.22
                    },
                    "dmlab": {
                        "adapter": 50.97,
                        "lora": 50.62,
                        "convpass": 51.93,
                        "convpass_attn": 51.88,
                        "fact_tt": 51.07,
                        "fact_tk": 52.39,
                        "vptshallow": 49.14,
                        "bitfit": 49.91,
                        "noah": 50.26
                    },
                    "kitti": {
                        "adapter": 78.10,
                        "lora": 78.93,
                        "convpass": 81.86,
                        "convpass_attn": 78.90,
                        "fact_tt": 77.19,
                        "fact_tk": 76.09,
                        "vptshallow": 79.80,
                        "bitfit": 75.11,
                        "noah": 79.06
                    },
                    "dsprites_loc": {
                        "adapter": 82.68,
                        "lora": 82.01,
                        "convpass": 86.72,
                        "convpass_attn": 84.29,
                        "fact_tt": 86.35,
                        "fact_tk": 86.19,
                        "vptshallow": 82.41,
                        "bitfit": 81.46,
                        "noah": 86.69
                    },
                    "dsprites_ori": {
                        "adapter": 54.48,
                        "lora": 54.30,
                        "convpass": 54.13,
                        "convpass_attn": 53.73,
                        "fact_tt": 53.25,
                        "fact_tk": 53.18,
                        "vptshallow": 54.63,
                        "bitfit": 51.58,
                        "noah": 53.84
                    },
                    "smallnorb_azi": {
                        "adapter": 35.16,
                        "lora": 37.18,
                        "convpass": 36.49,
                        "convpass_attn": 36.48,
                        "fact_tt": 36.84,
                        "fact_tk": 38.30,
                        "vptshallow": 34.33,
                        "bitfit": 37.11,
                        "noah": 34.91
                    },
                    "smallnorb_ele": {
                        "adapter": 43.16,
                        "lora": 43.08,
                        "convpass": 45.79,
                        "convpass_attn": 43.18,
                        "fact_tt": 42.55,
                        "fact_tk": 43.08,
                        "vptshallow": 42.88,
                        "bitfit": 37.47,
                        "noah": 42.81
                    }
                    }

    elif args.task == 'fgvc':
        datasets = ["CUB_200_2011", "nabirds", "OxfordFlower", "StanfordCars", "StanfordDogs"]

        data_dict = {
                    "CUB_200_2011": {
                        "adapter": 88.81,
                        "lora": 89.65,
                        "convpass": 88.99,
                        "convpass_attn": 88.76,
                        "fact_tt": 87.37,
                        "fact_tk": 87.12,
                        "vptshallow": 89.57,
                        "bitfit": 88.51,
                        "noah": 89.59
                    },
                    "nabirds": {
                        "adapter": 86.11,
                        "lora": 86.95,
                        "convpass": 84.21,
                        "convpass_attn": 86.94,
                        "fact_tt": 84.59,
                        "fact_tk": 83.47,
                        "vptshallow": 85.16,
                        "bitfit": 84.19,
                        "noah": 86.82
                    },
                    "OxfordFlower": {
                        "adapter": 98.42,
                        "lora": 99.19,
                        "convpass": 98.13,
                        "convpass_attn": 99.21,
                        "fact_tt": 99.05,
                        "fact_tk": 98.73,
                        "vptshallow": 99.11,
                        "bitfit": 98.91,
                        "noah": 99.06
                    },
                    "StanfordCars": {
                        "adapter": 86.21,
                        "lora": 84.39,
                        "convpass": 85.61,
                        "convpass_attn": 85.19,
                        "fact_tt": 85.48,
                        "fact_tk": 85.92,
                        "vptshallow": 84.77,
                        "bitfit": 84.04,
                        "noah": 85.14
                    },
                    "StanfordDogs": {
                        "adapter": 91.92,
                        "lora": 91.04,
                        "convpass": 92.82,
                        "convpass_attn": 93.12,
                        "fact_tt": 91.01,
                        "fact_tk": 92.49,
                        "vptshallow": 93.48,
                        "bitfit": 92.81,
                        "noah": 93.22
                    }
                }

    
        
    techniques = ["adapter", "lora",  "convpass", "convpass_attn", "fact_tt", "fact_tk","vptshallow","bitfit","noah"]
    metric = args.metric
    
    kendall_results = compute_kendall_tau(
        datasets, techniques, metric, data_dict, args
    )

