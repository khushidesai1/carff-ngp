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

latent_dim = 8
hidden_dim = 512
batch_size = 128
num_epochs = 500
lr = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
town = 3
K = 2

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

path = "./town3_video_merged/transforms_train.json"

f = open(path, "r")
data = json.load(f)

latents = {}
mus = {}
log_vars = {}
for frame in data["frames"]:
    # latents
    if frame["scene_id"] in latents.keys():
        latents[frame["scene_id"]].append(frame["latents"])
    else:
        latents[frame["scene_id"]] = [frame["latents"]]
    # mu
    if frame["scene_id"] in mus.keys():
        mus[frame["scene_id"]].append(frame["mu"])
    else:
        mus[frame["scene_id"]] = [frame["mu"]]
    # var
    if frame["scene_id"] in log_vars.keys():
        log_vars[frame["scene_id"]].append(frame["var"])
    else:
        log_vars[frame["scene_id"]] = [frame["var"]]



