import os
import json
import numpy as np

def create_combined_json(t, k, c):
    folder_path = "./Diffusion_score"
    backbone_path = os.path.join(folder_path, "backbone", "diffusion")
    peft_path = os.path.join(folder_path, "peft", "diffusion")
    
    datasets = [
        "caltech101", "cifar", "dtd", "oxford_flowers102", "oxford_iiit_pet", 
        "svhn", "sun397", "patch_camelyon", "eurosat", "resisc45", 
        "diabetic_retinopathy", "clevr_count", "clevr_dist", "dmlab", "kitti", 
        "dsprites_loc", "dsprites_ori", "smallnorb_azi", "smallnorb_ele"
    ]
    
    techniques_map = {
        "Adapter": "ADAPTER",
        "LoRA": "LORA",
        "Convpass": "convpass",
        "Convpass_attn": "convpassattn",
        "FacT-TT<16": "facttt",
        "FacT-TK32": "facttk",
        "VPT-Deep": "VPT",
        "BitFit": "bitfit",
        "NOAH": "NOAH"
    }

    
    
    for dataset in datasets:
        backbone_file_inter = os.path.join(backbone_path, f"{dataset}_{t}_{k}_{c}_inter_metrics.json")
        backbone_file_intra = os.path.join(backbone_path, f"{dataset}_{t}_{k}_{c}_intra_metrics.json")
        peft_file_inter = os.path.join(peft_path, f"{dataset}_{t}_{k}_{c}_inter_metrics.json")
        peft_file_intra = os.path.join(peft_path, f"{dataset}_{t}_{k}_{c}_intra_metrics.json")
        
        if not (os.path.exists(backbone_file_inter) and os.path.exists(backbone_file_intra) and os.path.exists(peft_file_inter) and os.path.exists(peft_file_intra)):
            continue
        
        with open(backbone_file_inter, "r") as f:
            backbone_inter_data = json.load(f)
        with open(backbone_file_intra, "r") as f:
            backbone_intra_data = json.load(f)
        with open(peft_file_inter, "r") as f:
            peft_inter_data = json.load(f)
        with open(peft_file_intra, "r") as f:
            peft_intra_data = json.load(f)
        
        inter_scores, intra_scores = [], []
        inter_score = {}
        intra_score = {}
        
        for tech in techniques_map.values():

            inter_score[tech] = float(peft_inter_data[tech]) - float(backbone_inter_data[tech])
            intra_score[tech] = float(peft_intra_data[tech]) - float(backbone_intra_data[tech])
            inter_scores.append(inter_score[tech])
            intra_scores.append(intra_score[tech])
        
        # Normalize inter_score (higher is better)
        inter_scores = np.array(inter_scores)
        inter_min, inter_max = inter_scores.min(), inter_scores.max()
        if inter_max - inter_min != 0:
            inter_scores = (inter_scores - inter_min) / (inter_max - inter_min)
        else:
            inter_scores = np.ones_like(inter_scores)
        
        # Normalize intra_score (lower is better, so we invert the normalized value)
        intra_scores = np.array(intra_scores)
        intra_min, intra_max = intra_scores.min(), intra_scores.max()
        if intra_max - intra_min != 0:
            intra_scores = 1 - (intra_scores - intra_min) / (intra_max - intra_min)
        else:
            intra_scores = np.ones_like(intra_scores)
        
        final_scores = {}
        for i, tech in enumerate(techniques_map.keys()):
            final_scores[techniques_map[tech]] = inter_scores[i] + intra_scores[i]
    

        output_path = f'scores/diffusion/{dataset}_metrics.json'

        os.makedirs("scores/diffusion", exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(final_scores, f, indent=4)
        
        print(f"Final scores saved to {output_path}")


if __name__ == "__main__":
    # Example parameter values.
    # For instance, if you have files like: caltech101_l2_10.0_20_32_inter_metrics.json,
    # then t = 10.0, k = 20, c = 32.
    ts = [10]
    ks = [20]
    n_c = [32]
    
    for t in ts:
        for k in ks:
            for c in n_c:
                create_combined_json(t, k, c)

