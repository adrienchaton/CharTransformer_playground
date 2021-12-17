# -*- coding: utf-8 -*-
"""
@author: Adrien Bitton

A set of data utilities to work with a dataset of names from different countries
data can be downloaded via this page https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

Character-level NLP
"""


import math
import os
import glob
import unicodedata
import string
from functools import partial
import json

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


def findFiles(path): return glob.glob(path)


def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)


def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]


def load_data_names(data_path, fixed_len_range=None, token_dict={'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}):
    # read all lines and characters
    all_letters = string.ascii_letters + " .,;'"
    category_lines = {}
    all_categories = []
    tot_lines = 0
    for filename in findFiles(os.path.join(data_path, '*.txt')):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename, all_letters)
        tot_lines += len(lines)
        category_lines[category] = lines
    # filter by lengths and check used characters
    unused_letters = [_letter for _letter in all_letters]
    used_letters = []
    max_len = 0
    min_len = float("inf")
    lengths = []
    for _cntry in category_lines:
        for _name in category_lines[_cntry].copy():
            if fixed_len_range is None or (len(_name) >= fixed_len_range[0] and len(_name) <= fixed_len_range[1]):
                max_len = np.max([max_len, len(_name)])
                min_len = int(np.min([min_len, len(_name)]))
                lengths.append(len(_name))
                for _letter in _name:
                    if _letter in unused_letters:
                        unused_letters.remove(_letter)
                    if _letter in all_letters and _letter not in used_letters:
                        used_letters.append(_letter)
            else:
                category_lines[_cntry].remove(_name)
    # assign character-level tokens
    used_letters = sorted(used_letters)
    for i, w in enumerate(used_letters):
        token_dict[w] = i + 4
    n_samples = np.sum([len(category_lines[_k]) for _k in category_lines])
    return category_lines, token_dict, n_samples, min_len, max_len, lengths


def tokenize_line(line, token_dict):
    # line is a string, output is a list of integers
    return [token_dict[_c] for _c in line]


def detokenize_line(line, token_dict, excluded_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]']):
    # line is a numpy array of integers, output is a string
    excluded_tokens = [token_dict[_c] for _c in excluded_tokens]
    detokenized = ""
    for _c in list(line):
        if _c not in excluded_tokens:
            detokenized += list(token_dict.keys()
                                )[list(token_dict.values()).index(_c)]
    return detokenized


def tokenize_chars(category_lines, token_dict, max_len):
    # also handles adding [CLS] and [PAD] tokens + assigning labels
    dataset_names = []
    dataset_labels = []
    label_dict = dict()
    for i, cntry in enumerate(list(category_lines.keys())):
        label_dict[i] = cntry
        for name in category_lines[cntry]:
            dataset_names.append([token_dict['[CLS]']]+tokenize_line(name,
                                 token_dict)+[token_dict['[PAD]']]*(max_len-len(name)))
            dataset_labels.append(i)
    dataset_names = np.stack(dataset_names)
    dataset_labels = np.array(dataset_labels)
    return dataset_names, dataset_labels, label_dict


def stratified_dataset(dataset_names, dataset_labels, valid_size=0.1, test_size=0.1):
    # splitting test set with balanced classes
    train_indices, test_indices = train_test_split(
        list(range(len(dataset_labels))), test_size=test_size, stratify=dataset_labels)
    train_data, train_labels = dataset_names[train_indices], dataset_labels[train_indices]
    test_data, test_labels = dataset_names[test_indices], dataset_labels[test_indices]
    # splitting train/valid sets with balanced classes
    train_indices, valid_indices = train_test_split(
        list(range(len(train_labels))), test_size=valid_size, stratify=train_labels)
    train_data, train_labels = dataset_names[train_indices], dataset_labels[train_indices]
    valid_data, valid_labels = dataset_names[valid_indices], dataset_labels[valid_indices]
    print("shapes for train_data,train_labels,valid_data,valid_labels,test_data,test_labels\n",
          train_data.shape, train_labels.shape, valid_data.shape, valid_labels.shape, test_data.shape, test_labels.shape)
    # creating tensor datasets
    train_dataset = TensorDataset(torch.from_numpy(
        train_data).long(), torch.from_numpy(train_labels).long())
    valid_dataset = TensorDataset(torch.from_numpy(
        valid_data).long(), torch.from_numpy(valid_labels).long())
    test_dataset = TensorDataset(torch.from_numpy(
        test_data).long(), torch.from_numpy(test_labels).long())
    return train_dataset, valid_dataset, test_dataset


def prepare_datasets_names(data_path, fixed_len_range=None, token_dict={'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}, valid_size=0.1, test_size=0.1):
    category_lines, token_dict, n_samples, min_len, max_len, lengths = load_data_names(
        data_path, fixed_len_range=fixed_len_range, token_dict=token_dict)
    dataset_names, dataset_labels, label_dict = tokenize_chars(
        category_lines, token_dict, max_len)
    train_dataset, valid_dataset, test_dataset = stratified_dataset(
        dataset_names, dataset_labels, valid_size=valid_size, test_size=test_size)
    return train_dataset, valid_dataset, test_dataset, token_dict, label_dict, max_len


# example of default collate_fn with configurable arguments "misc"
def default_collate_fn(data, misc):
    # batching function to benefit from multiple workers in the dataloader
    # data is a list of size batch_size
    # each element is a tuple of tensors e.g. input,target
    print(misc)
    mb_dict = dict()
    data_tensors, data_labels = zip(*data)
    data_tensors = torch.stack(data_tensors)
    data_labels = torch.stack(data_labels)
    mb_dict["data_tensors"] = data_tensors
    mb_dict["data_labels"] = data_labels
    return mb_dict


def custom_collate_fn(data, training_objectives, token_dict, n_special_tokens, max_len, mask_val, random_masking_prob, random_splitting_prob):
    with torch.no_grad():
        # batching function to benefit from multiple workers in the dataloader
        # data is a list of size batch_size
        # each element is a tuple of tensors e.g. input,target
        mb_dict = dict()
        padded_data, cntry_labels = zip(*data)
        # shape [batch_size,max_len+1] including [CLS] and [PAD] tokens
        padded_data = torch.stack(padded_data)
        cntry_labels = torch.stack(cntry_labels)
        batch_size = padded_data.shape[0]

        # padding mask for e.g. sequence classification task
        padding_mask = torch.where(padded_data == token_dict['[PAD]'], torch.ones_like(
            padded_data), torch.zeros_like(padded_data))
        # padding_mask = padding_mask.unsqueeze(1).repeat(1,max_len+1,1)
        # padding_mask is left of shape [batch_size,max_len+1] == src_key_padding_mask

        mb_dict["padded_data"], mb_dict["padding_mask"] = padded_data, padding_mask
        if "CLS" in training_objectives:
            mb_dict["cntry_labels"] = cntry_labels

        if "MLM" in training_objectives:
            # random masking some regular tokens and input to the model so that it predicts the masked tokens
            masked_data = padded_data.clone()
            for _i in range(batch_size):
                reg_tokens = torch.where(
                    (masked_data[_i] >= n_special_tokens).float() == 1)[0]
                rand_mask = (torch.rand(len(reg_tokens)) <
                             random_masking_prob).float()
                if torch.sum(rand_mask) >= 1 and torch.sum(rand_mask) < len(reg_tokens):
                    masked_data[_i][reg_tokens[torch.where(
                        rand_mask == 1)[0]]] = token_dict['[MASK]']
                else:
                    rand_mask = torch.randint(len(reg_tokens), (1, 1))
                    masked_data[_i][reg_tokens[rand_mask]
                                    ] = token_dict['[MASK]']
            masking_pos = torch.where(
                masked_data.view(-1) == token_dict['[MASK]'])[0]
            # for evaluating the loss
            # masking_inputs = masked_data.view(-1)[masking_pos]
            # masking_targets = padded_data.view(-1)[masking_pos]

            masking_mask = torch.where(masked_data == token_dict['[MASK]'], torch.ones_like(
                masked_data), torch.zeros_like(masked_data))
            # masking_mask = masking_mask.unsqueeze(1).repeat(1,max_len+1,1)
            # masking_mask is left of shape [batch_size,max_len+1] == src_key_padding_mask

            # combined with previous padding mask
            masking_mask = ((padding_mask+masking_mask) > 0).float()
            # TODO: for the masking task, do we use padding_mask alone or also masking_mask ?
            # and swap the masked tokens with either another random token or the true token
            if np.random.rand() > 0.85:
                if np.random.rand() > 0.5:
                    masked_data[torch.where(masked_data == token_dict['[MASK]'])] = padded_data[torch.where(
                        masked_data == token_dict['[MASK]'])]
                else:
                    masked_data[torch.where(masked_data == token_dict['[MASK]'])] = np.random.choice(
                        len(token_dict)-n_special_tokens)+n_special_tokens

            mb_dict["masked_data"], mb_dict["masking_pos"], mb_dict["masking_mask"] = masked_data, masking_pos, masking_mask

        if "LA" in training_objectives:
            # next token prediction with causal look-ahead masking

            # lookahead_mask = torch.triu(torch.ones(max_len+1, max_len+1), diagonal=1).unsqueeze(0).repeat(batch_size,1,1) # upper triangular
            # lookahead_mask = ((padding_mask+lookahead_mask)>0).float() #  combined with previous padding mask
            # lookahead_mask = lookahead_mask[:,1:-1,1:-1] # bptt_len = max_len-1

            lookahead_src_mask = torch.triu(
                torch.ones(max_len-1, max_len-1), diagonal=1)
            # lookahead_src_mask is left of shape [max_len-1,max_len-1] == src_mask
            lookahead_key_mask = padding_mask[:, 1:-1]
            # lookahead_key_mask is of shape [batch_size,max_len-1] == src_key_padding_mask

            # remove [CLS] and last position
            lookahead_inputs = padded_data.clone()[:, 1:-1]
            # remove [CLS] and first position
            lookahead_targets = padded_data.clone()[:, 2:]
            # [PAD] tokens appear first in the target sequence
            lookahead_pos = torch.where(
                lookahead_targets.reshape(-1) != token_dict['[PAD]'])[0]
            # for evaluating the loss
            # lookahead_inputs = lookahead_inputs.reshape(-1)[lookahead_pos]
            # lookahead_targets = lookahead_targets.reshape(-1)[lookahead_pos]
            mb_dict["lookahead_inputs"], mb_dict["lookahead_targets"], mb_dict["lookahead_pos"], mb_dict["lookahead_src_mask"], mb_dict[
                "lookahead_key_mask"] = lookahead_inputs, lookahead_targets, lookahead_pos, lookahead_src_mask, lookahead_key_mask

        if "NSP" in training_objectives:
            # random split and concatenation (~ next sentence prediction with '[SEP]' token)
            # the NSP task is usually computed on [CLS] masked sentence A [SEP] masked sentence B [SEP]
            # here we do [CLS] first half [SEP] second half with 50% chance next / not next
            # [CLS] first half [SEP] + NSP_embedding_A / second half + NSP_embedding_B
            nsp_data = []
            nsp_posA = []
            nsp_labels = (torch.rand((batch_size)) <
                          random_splitting_prob).long()  # 1 is ~ not next
            for _i in range(batch_size):
                reg_tokens = torch.where(
                    (padded_data[_i] >= n_special_tokens).float() == 1)[0]
                # min_len >= 2 ; [SEP] is neither start or end
                rand_sep = torch.randint(len(reg_tokens)-1, (1, 1))
                rand_sep = reg_tokens[rand_sep+1]
                rand_posA = torch.zeros((max_len+2))
                rand_posA[:rand_sep+1] = 1
                if nsp_labels[_i] == 0:
                    # add a random [SEP] without shuffling
                    nsp_tensor = [padded_data[_i][:rand_sep], torch.tensor(
                        [token_dict['[SEP]']]), padded_data[_i][rand_sep:]]
                    nsp_data.append(torch.cat(nsp_tensor))
                else:
                    # add a random [SEP] and pad with random other sequence
                    nsp_tensor = [padded_data[_i][:rand_sep],
                                  torch.tensor([token_dict['[SEP]']])]
                    rand_not_next = list(range(batch_size))
                    rand_not_next.remove(_i)
                    rand_not_next = np.random.choice(rand_not_next)
                    reg_tokens = torch.where(
                        (padded_data[rand_not_next] >= n_special_tokens).float() == 1)[0]
                    # min_len >= 2 ; [SEP] is neither start or end
                    rand_sep = torch.randint(len(reg_tokens)-1, (1, 1))
                    rand_sep = reg_tokens[rand_sep+1]
                    nsp_tensor = torch.cat(
                        nsp_tensor+[padded_data[rand_not_next][rand_sep:]])[:max_len+2]
                    if len(nsp_tensor) < (max_len+2):
                        nsp_tensor = torch.cat([nsp_tensor, torch.tensor(
                            [token_dict['[PAD]']]*(max_len+2-len(nsp_tensor)))])
                    nsp_data.append(nsp_tensor)
                nsp_posA.append(rand_posA)
            nsp_data = torch.stack(nsp_data)
            nsp_posA = torch.stack(nsp_posA).long()

            nsp_mask = torch.where(nsp_data == token_dict['[PAD]'], torch.ones_like(
                nsp_data), torch.zeros_like(nsp_data))
            # nsp_mask = nsp_mask.unsqueeze(1).repeat(1,max_len+2,1) # the nsp padding mask has size max_len+2 ('[CLS]' and '[SEP]' tokens)
            # nsp_mask is left of shape [batch_size,max_len+2] == src_key_padding_mask

            mb_dict["nsp_data"], mb_dict["nsp_labels"], mb_dict["nsp_posA"], mb_dict["nsp_mask"] = nsp_data, nsp_labels, nsp_posA, nsp_mask

        # *mask_val for all masks, to add to attention logits before softmax (~ 0 attention weights)
        # for _k in mb_dict:
        #     if _k.endswith("_mask"):
        #         mb_dict[_k] = mb_dict[_k]*mask_val

        for _k in mb_dict:
            if _k.endswith("_mask"):
                mb_dict[_k] = mb_dict[_k].bool()

        return mb_dict


if __name__ == "__main__":
    data_path = "data_names"
    valid_size = 0.1
    test_size = 0.1

    # fixed_len_range = None
    fixed_len_range = [4, 11]

    training_objectives = ["CLS", "MLM", "LA", "NSP"]
    batch_size = 2  # NSP requires batch_size>1

    # training_objectives = ["CLS","MLM","LA"]
    # batch_size = 1

    token_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
                  '[MASK]': 3}  # special tokens used for training
    n_special_tokens = len(token_dict)
    mask_val = -1e9
    random_masking_prob = 0.15  # masked LM
    random_splitting_prob = 0.5  # ~ next sentence prediction

    # category_lines,token_dict,n_samples,min_len,max_len,lengths = load_data_names(data_path,fixed_len_range=fixed_len_range,token_dict=token_dict)
    # plt.figure()
    # sns.histplot(lengths)
    # dataset_names,dataset_labels,label_dict = tokenize_chars(category_lines,token_dict,max_len)
    # train_dataset,valid_dataset,test_dataset = stratified_dataset(dataset_names,dataset_labels,valid_size=valid_size,test_size=test_size)

    train_dataset, valid_dataset, test_dataset, token_dict, label_dict, max_len = prepare_datasets_names(
        data_path, fixed_len_range=fixed_len_range, token_dict=token_dict, valid_size=valid_size, test_size=test_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  num_workers=0, collate_fn=partial(default_collate_fn, misc=1))  # issue with multiple workers ...
    for i, mb_dict in enumerate(train_dataloader):
        print(mb_dict["data_tensors"].shape, mb_dict["data_labels"].shape)
        break

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                                  collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                     mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))  # issue with multiple workers ...
    for i, mb_dict in enumerate(train_dataloader):
        padded_data, padding_mask = mb_dict["padded_data"], mb_dict["padding_mask"]
        if "CLS" in training_objectives:
            cntry_labels = mb_dict["cntry_labels"]
        if "MLM" in training_objectives:
            masked_data, masking_pos, masking_mask = mb_dict[
                "masked_data"], mb_dict["masking_pos"], mb_dict["masking_mask"]
        if "LA" in training_objectives:
            lookahead_inputs, lookahead_targets, lookahead_pos, lookahead_src_mask, lookahead_key_mask = mb_dict["lookahead_inputs"], mb_dict[
                "lookahead_targets"], mb_dict["lookahead_pos"], mb_dict["lookahead_src_mask"], mb_dict["lookahead_key_mask"]
        if "NSP" in training_objectives:
            nsp_data, nsp_labels, nsp_posA, nsp_mask = mb_dict["nsp_data"], mb_dict[
                "nsp_labels"], mb_dict["nsp_posA"], mb_dict["nsp_mask"]
        break

    # detokenize and check all tensors are correct

    print("detokenize padded_data")
    print(detokenize_line(
        padded_data[0].numpy(), token_dict, excluded_tokens=[]))
    print(detokenize_line(padded_data[0].numpy(), token_dict, excluded_tokens=[
          '[PAD]', '[CLS]', '[SEP]', '[MASK]']))
    print(padding_mask[0, :])  # -inf on padding columns
    if "MLM" in training_objectives:
        print("detokenize masked_data")
        print(detokenize_line(
            masked_data[0].numpy(), token_dict, excluded_tokens=[]))
        print(masking_mask[0, :])  # -inf on masking column and padding columns
        print(masking_pos)
    if "LA" in training_objectives:
        print("detokenize lookahead data")
        print(detokenize_line(
            lookahead_inputs[0].numpy(), token_dict, excluded_tokens=[]))
        print(detokenize_line(
            lookahead_targets[0].numpy(), token_dict, excluded_tokens=[]))
        print(lookahead_pos)
        print("lookahead_src_mask")
        print(lookahead_src_mask[0, :])  # -inf on upper triangula
        print(lookahead_src_mask[-1, :])
        print("lookahead_key_mask")
        print(lookahead_key_mask[0, :])  # padding columns
    if "NSP" in training_objectives:
        print("detokenize nsp_data")
        print(detokenize_line(nsp_data[0].numpy(),
              token_dict, excluded_tokens=[]))
        print(nsp_labels[0])
        print(nsp_posA[0])
        print(nsp_mask[0, :])  # -inf on padding columns
