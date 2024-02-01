import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import tqdm
from sklearn.manifold import TSNE
import os

import torch
import argparse
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *
from functools import partial
from loss import huber_loss
from init import init
import sys

from nerf.network_tcnn import NeRFNetwork
from sklearn.cluster import KMeans

# ngp training
opt = init('./town3_video_merged/')
seed_everything(0)
assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"

model = NeRFNetwork(
            encoding="hashgrid",
                bound=opt.bound,
                    cuda_ray=opt.cuda_ray,
                        density_scale=1,
                            min_near=opt.min_near,
                                density_thresh=opt.density_thresh,
                                    bg_radius=opt.bg_radius,
                                    )

criterion = torch.nn.MSELoss(reduction='none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
train_loader = NeRFDataset(opt, device=device, type='train').dataloader()
scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=50)
gui = NeRFGUI(opt, trainer, train_loader)

gui.render(2100) # 500 to reach 0.55 progress

def check_densities():
    max_density = -1
    timestamp = -1

    for i in range(6):
        density = gui.calculate_densities(i)
        if density > max_density:
            max_density = density
            timestamp = i
        # print("Timestamp", i, "density:", density)

    print("--------------------------------")
    print("Predicted timestamp:", timestamp)
    print("Density value:", max_density)
    print("--------------------------------")

for i in range(6):
    gui.update_latent_from_image(f'v2-t{i}')

    gui.render_bev(dest_path=f'rendered_t{i}.png', color_t=0)
    check_densities()

"""
accuracies = []
num_samples = []
for n in range(1, 51):
    print("Num samples:", n)
    rand_t = np.random.choice(a=[1, 4], p=[0.5, 0.5])
    acc = gui.predict_probe_n(f"train/cam-v2-t{rand_t}", n)
    print("------------------------")
    print("Accuracy:", acc)
    print("------------------------")
    accuracies.append(acc)
    num_samples.append(n)

print(num_samples)
print(accuracies)
"""
