"""
@author: Adrien Bitton

Evaluation script of transformer encoder for character-level NLP
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

from data_names_utils import prepare_datasets_names, custom_collate_fn, plot_tb_logs,detokenize_line
from model import TransformerPredictor, LoggingTestLossCallback


###############################################################################
# model arguments

parser = ArgumentParser()
parser.add_argument('--gpu_dev', default="", type=str)
parser.add_argument('--mname', default="test1_4losses", type=str)
parser.add_argument('--data_path', default="data_names", type=str)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--fp16', action='store_true')
args = parser.parse_args()

eval_split = "train"
batch_size = 4 # NSP requires batch_size>1
shuffle = True
num_workers = 0
pin_memory = False

gpu_dev = args.gpu_dev
mname = args.mname
data_path = args.data_path

curr_dir = os.getcwd()
mdir = os.path.join(curr_dir, "training_runs", mname)
print("loading checkpoint from mdir", mdir)

config = glob.glob(os.path.join(mdir, "*config*.json"))[0]
print("loading config", config)

with open(config) as json_file:
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


###############################################################################
# compute settings with single GPU support

pl.seed_everything(1234)
np.random.seed(1234)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_dev
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpus = 1 if torch.cuda.is_available() else 0
precision = 16 if args.fp16 else 32

if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.benchmark = True

print("\ndevice", device)


###############################################################################
# data preparation and last model checkpoint restoring

n_special_tokens = len(token_dict)

train_dataset, valid_dataset, test_dataset, token_dict, label_dict, max_len = prepare_datasets_names(
    data_path, fixed_len_range=fixed_len_range, token_dict=token_dict, valid_size=valid_size, test_size=test_size)

n_classes = len(label_dict)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers,pin_memory=pin_memory,
                              collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                 mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))

valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers,pin_memory=pin_memory,
                              collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                 mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers,pin_memory=pin_memory,
                             collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))


###############################################################################
# sample evaluation data

if shuffle:
    # the previous seeds are set to get correct train/valid/test splits
    # the seeds are randomly reset for evaluating on random samples
    np.random.seed(int(time.time()))
    pl.seed_everything(int(time.time()))

if eval_split=="train":
    dloader = train_dataloader
if eval_split=="valid":
    dloader = valid_dataloader
if eval_split=="test":
    dloader = test_dataloader

for _, mb_dict in enumerate(dloader):
    break


###############################################################################
# last model checkpoint restoring

# random initialization model
model_untrained = TransformerPredictor(len(token_dict), model_dim, input_dropout, max_len+2, training_objectives, lamb_CLS, detach_CLS,
                             E_position, nn_act, T_n_head, T_hidden_dim, T_dropout, T_norm_first, T_n_layers,
                             output_dropout, cls_mode, cls_masked, token_pred_transposed, n_classes=n_classes,
                             optim_model=None,lr=None,weight_decay=None, warmup=None, max_iters=None)
model_untrained.to(device)
model_untrained.eval()

# trained model
ckpt_file = glob.glob(os.path.join(mdir, "checkpoints", "*.ckpt"))[0]
yaml_file = glob.glob(os.path.join(mdir, "*.yaml"))[0]
print("loading ckpt_file,yaml_file", ckpt_file,yaml_file)
model = TransformerPredictor.load_from_checkpoint(checkpoint_path=ckpt_file,hparams_file=yaml_file,map_location='cpu')
model.to(device)
model.eval()

print("\nmodel running on device", model.device)
print("\nmodel hyper-parameters", model.hparams)


###############################################################################
# visualize self-attention maps

def plot_attention_maps(suptitle,model,mb_dict,i_sample,sel_maps,num_heads,num_layers,token_dict):
    attn_logits_maps,attn_probs_maps = model.extract_selfattention_maps(mb_dict["padded_data"],
                mask=None, src_key_padding_mask=mb_dict["padding_mask"], add_positional_encoding=True, add_segment_embedding=None)
    if sel_maps=="logits":
        suptitle += "attn_logits_maps"
        maps = attn_logits_maps
    if sel_maps=="probs":
        suptitle += "attn_probs_maps"
        maps = attn_probs_maps
    seq_len = mb_dict["padded_data"].shape[1]
    input_data = detokenize_line(mb_dict["padded_data"][i_sample].cpu().numpy(), token_dict, excluded_tokens=[])
    for _t in ['[PAD]', '[CLS]', '[SEP]', '[MASK]']:
        input_data = input_data.replace(_t,"$")
    input_data = [_c for _c in input_data]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(maps[row][i_sample,column,:,:].cpu().numpy(), origin="lower", vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data)
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data)
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
    fig.subplots_adjust(hspace=0.5)
    plt.suptitle(suptitle)
    plt.show()

# for phase in [["trained model ",model],["untrained model ",model_untrained]]:
#     for sel_maps in ["logits","probs"]:
#         for i in range(batch_size):
#             plot_attention_maps(phase[0],phase[1],mb_dict,i,sel_maps,T_n_head,T_n_layers,token_dict)

# for phase in [["trained model ",model]]:
#     for sel_maps in ["probs"]:
#         for i in range(batch_size):
#             plot_attention_maps(phase[0],phase[1],mb_dict,i,sel_maps,T_n_head,T_n_layers,token_dict)


###############################################################################
# export text processing (reconstruction and classification)

with torch.no_grad():
    _, mb_dict, _ = model.calculate_losses(mb_dict)

for i in range(batch_size):
    print("\n*** processing sample",i)
    print(detokenize_line(mb_dict["padded_data"][i].cpu().numpy(), token_dict, excluded_tokens=[]))
    for t_obj in training_objectives:
        # 'CLS', 'MLM', 'LA', 'NSP'
        if t_obj=="CLS":
            pred_label = label_dict[np.argmax(mb_dict["pred_labels"][i].cpu().numpy())]
            true_label = label_dict[mb_dict["cntry_labels"][i].item()]
            print("predicted and true country labels",pred_label,true_label)


###############################################################################
# plot e.g. embedding scatter
# TODO: compare before and after training






