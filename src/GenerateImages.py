from DiffusionFreeGuidence.TrainCondition import train, eval
import torch

import os
from typing import Dict
from torchvision.utils import save_image

from DiffusionFreeGuidence.DiffusionCondition import (
    GaussianDiffusionSampler
)
from DiffusionFreeGuidence.ModelCondition import UNet

from PIL import Image
import time

torch.manual_seed(1)

def eval(modelConfig: Dict, idx):
    device = torch.device(modelConfig["device"])
    # load model and evaluate
    with torch.no_grad():
        step = int(modelConfig["batch_size"] // 10)
        labelList = []
        k = 0
        for i in range(1, modelConfig["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print("labels: ", labels)
        model = UNet(
            T=modelConfig["T"],
            num_labels=10,
            ch=modelConfig["channel"],
            ch_mult=modelConfig["channel_mult"],
            num_res_blocks=modelConfig["num_res_blocks"],
            dropout=modelConfig["dropout"],
        ).to(device)
        ckpt = torch.load(
            os.path.join(modelConfig["save_dir"], modelConfig["test_load_weight"]),
            map_location=device,
        )
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model,
            modelConfig["beta_1"],
            modelConfig["beta_T"],
            modelConfig["T"],
            w=modelConfig["w"],
        ).to(device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[
                modelConfig["batch_size"],
                3,
                modelConfig["img_size"],
                modelConfig["img_size"],
            ],
            device=device,
        )

        sampledImgs = sampler(noisyImage, labels)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        # print(sampledImgs)
        t = time.time()
        for i, img in enumerate(sampledImgs):
            ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(os.path.join(modelConfig["sampled_dir"], f"{(idx + i)}".zfill(4) + ".png"))
        print(f"Time to save: {time.time() - t}")

def main(model_config=None):
    modelConfig = {
        "state": "eval", 
        "epoch": 70,
        "batch_size": 80,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.028,
        "img_size": 32,
        "grad_clip": 1.0,
        "device": "cuda:0",
        "w": 1.8,
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "DiffusionConditionWeight.pt",
        "sampled_dir": "./GeneratedImgs/",
        "sampledNoisyImgName": "NoisyGuidenceImgs.png",
        "sampledImgName": "SampledGuidenceImgs.png",
        "nrow": 8,
    }
    if model_config is not None:
        modelConfig = model_config
    
    eval(modelConfig, 1)
    
    # for i in range(0, 5000, modelConfig["batch_size"]):
    #     eval(modelConfig, i)


if __name__ == "__main__":
    main()
