"""
@author: Adrien Bitton

Training script of transformer encoder for character-level NLP
"""


import math
import os
import glob
import unicodedata
import string
from functools import partial
import json
from argparse import ArgumentParser
import shutil
import time

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data_names_utils import prepare_datasets_names, custom_collate_fn, plot_tb_logs
from model import TransformerPredictor, LoggingTestLossCallback


###############################################################################
# training arguments

# TODO: check some launching issues with multi-processing and CUDA ?
# RuntimeError: CUDA error: unknown error
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

# python train.py --gpu_dev 1 --mname test0_4losses --max_iters 40000 --config train_4losses_config0.json --num_workers 4 --pin_memory
# python train.py --gpu_dev 1 --mname test0_4losses --max_iters 40000 --config train_4losses_config0.json --num_workers 0 --pin_memory
# CUDA_LAUNCH_BLOCKING=1 python train.py --gpu_dev 1 --mname test0_4losses --max_iters 40000 --config train_4losses_config0.json --num_workers 4 --pin_memory

parser = ArgumentParser()
parser.add_argument('--gpu_dev', default="", type=str)
parser.add_argument('--mname', default="a_model", type=str)
parser.add_argument('--data_path', default="data_names", type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--optim_model', default="adam", type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--warmup', default=250, type=int)
parser.add_argument('--max_iters', default=100000, type=int)
parser.add_argument('--gradient_clip_val', default=3., type=float)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--pin_memory', action='store_true')
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--config', default="train_4losses_config0.json", type=str)
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--profiler', action='store_true')
parser.add_argument('--early_stop_patience', default=10, type=int)
args = parser.parse_args()

# training hyper-parameters

gpu_dev = args.gpu_dev
mname = args.mname
data_path = args.data_path
batch_size = args.batch_size
optim_model = args.optim_model # either "adam" or "adamw"
lr = args.lr
weight_decay = args.weight_decay
warmup = args.warmup # for lr, in steps
max_iters = args.max_iters
gradient_clip_val = args.gradient_clip_val
num_workers = args.num_workers
pin_memory = args.pin_memory
deterministic = args.deterministic
config = args.config
profiler = args.profiler
early_stop_patience = args.early_stop_patience # in epochs

curr_dir = os.getcwd()
default_root_dir = os.path.join(curr_dir, "training_runs", mname)
print("writing outputs into default_root_dir", default_root_dir)
# lighting is writting output files in default_root_dir/lightning_logs/version_0/
tmp_dir = os.path.join(default_root_dir, "lightning_logs", "version_0")

with open(os.path.join("configs", config)) as json_file:
    config = json.load(json_file)

valid_size = config["valid_size"]
test_size = config["test_size"]
fixed_len_range = config["fixed_len_range"] # restrict the data min/max lengths
training_objectives = config["training_objectives"] # choose which objectives to train on
lamb_CLS = config["lamb_CLS"] # multiplicative factor on the supervised CLS loss
detach_CLS = config["detach_CLS"] # detaching means CLS loss does not optimize the transformer encoder
token_dict = config["token_dict"]
mask_val = config["mask_val"]
random_masking_prob = config["random_masking_prob"] # for MLM
random_splitting_prob = config["random_splitting_prob"] # for NSP

# model hyper-parameters

model_dim = config["model_dim"]
input_dropout = config["input_dropout"]
E_position = config["E_position"] # either sinusoidal or learned position embedding
nn_act = config["nn_act"]
T_n_head = config["T_n_head"]
T_hidden_dim = config["T_hidden_dim"]
T_dropout = config["T_dropout"]
T_norm_first = config["T_norm_first"] # LN first can help stabilizing the early training
T_n_layers = config["T_n_layers"]
output_dropout = config["output_dropout"]
cls_mode = config["cls_mode"] # either using start CLS position or averaging all encoder outputs
cls_masked = config["cls_masked"] # performing CLS prediction from masked or unmasked data
token_pred_transposed = config["token_pred_transposed"] # character prediction, either using linear layers or transposed of embedding matrix

print("\ntraining arguments", vars(args))


###############################################################################
# compute settings with single GPU support

pl.seed_everything(1234)
np.random.seed(1234)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = 1 if torch.cuda.is_available() else 0
precision = 16 if args.fp16 else 32

if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.benchmark = True

print("\ndevice", device)


###############################################################################
# data  preparation

n_special_tokens = len(token_dict)

train_dataset, valid_dataset, test_dataset, token_dict, label_dict, max_len = prepare_datasets_names(
    data_path, fixed_len_range=fixed_len_range, token_dict=token_dict, valid_size=valid_size, test_size=test_size)

n_classes = len(label_dict)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,pin_memory=pin_memory,
                              collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                 mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))

valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers,pin_memory=pin_memory,
                              collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                 mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers,pin_memory=pin_memory,
                             collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))

model = TransformerPredictor(len(token_dict), model_dim, input_dropout, max_len+2, training_objectives, lamb_CLS, detach_CLS,
                             E_position, nn_act, T_n_head, T_hidden_dim, T_dropout, T_norm_first, T_n_layers,
                             output_dropout, cls_mode, cls_masked, token_pred_transposed, n_classes=n_classes,
                             optim_model=optim_model,lr=lr,weight_decay=weight_decay, warmup=warmup, max_iters=max_iters)

model.to(device)
print("\nmodel running on device", model.device)
print("\nmodel hyper-parameters", model.hparams)

model.test_dataloader = test_dataloader

model.train()
for _, mb_dict in enumerate(train_dataloader):
    model.gradient_check(mb_dict)
    break


###############################################################################
# training

time.sleep(10)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
test_monitor = LoggingTestLossCallback()
early_stop_callback = EarlyStopping(monitor="valid_tot_loss", min_delta=0.00, patience=early_stop_patience,
                                    verbose=True, mode="min")  # early_stop_patience == epochs patience with no decrease

trainer = pl.Trainer(max_steps=max_iters, check_val_every_n_epoch=1,
                     gpus=gpus, precision=precision, benchmark=True,
                     default_root_dir=default_root_dir, profiler=profiler,
                     progress_bar_refresh_rate=50, callbacks=[lr_monitor, test_monitor, early_stop_callback])

trainer.fit(model, train_dataloader, valid_dataloader)


###############################################################################
# evaluation

# TODO: make evaluation functions e.g. write text files for each dataset split
# with detokenized inputs and predictions + visualization of attentions

# train_result = trainer.test(model, test_dataloaders=train_dataloader, verbose=False)
# valid_result = trainer.test(model, test_dataloaders=valid_dataloader, verbose=False)
# test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)

model.to(device)
model.eval()


###############################################################################
# misc.

shutil.move(tmp_dir, os.path.join(curr_dir, "training_runs"))
shutil.rmtree(default_root_dir)
os.rename(os.path.join(curr_dir, "training_runs", "version_0"), default_root_dir)

plot_tb_logs(default_root_dir,training_objectives)
shutil.copyfile(os.path.join("configs", args.config), os.path.join(default_root_dir, args.config))
with open(os.path.join(default_root_dir, 'argparse.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)
