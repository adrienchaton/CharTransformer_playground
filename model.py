"""
@author: Adrien Bitton

Transformer encoder for character-level NLP
the convention used is batch_first=True i.e. [batch_size,seq_len,n_features]
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
# import torchmetrics


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    # to avoid training instabilities, e.g.
    # optimizer = optim.Adam([p], lr=1e-3)
    # lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)
    # initially, lr=0 ; after warmup lr_scheduler.step() lr=1e-3 ; after max_iters lr_scheduler.step() lr~0
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class OutputLinears(nn.Module):
    # to use on the output states of the transformer encoder
    # e.g. num_classes = 2 for NSP
    # e.g. num_classes = n_tokens for MLM or next token prediction
    # to use with F.cross_entropy (Softmax/CE)
    def __init__(self, model_dim, num_classes, p_dropout, nn_act):
        super().__init__()
        if nn_act == "relu":
            activation = nn.ReLU(inplace=True)
        if nn_act == "gelu":
            activation = nn.GELU()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            activation,
            nn.Dropout(p_dropout),
            nn.Linear(model_dim, num_classes))

    def forward(self, x):
        return self.layers(x)


class PositionalEncoding(nn.Module):
    # input_dim is the embedding_size
    # max_len is the longest sequence length to receive
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class LearnedPositionalEncoding(nn.Module):
    # input_dim is the embedding_size
    # max_len is the longest sequence length to receive
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        self.pe = nn.Parameter(torch.randn(max_len, model_dim).unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerPredictor(pl.LightningModule):
    # a half-way configurable Transformer class derived from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # full built-in class available at https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
    # full custom class available at https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
    def __init__(self, n_tokens, model_dim, input_dropout, max_len, training_objectives, lamb_CLS, detach_CLS,
                 E_position, nn_act, T_n_head, T_hidden_dim, T_dropout, T_norm_first, T_n_layers,
                 output_dropout, cls_mode, cls_masked, token_pred_transposed, n_classes=None,
                 optim_model="adam",lr=2e-4,weight_decay=1e-2, warmup=500, max_iters=100000):
        super().__init__()
        assert model_dim % T_n_head == 0
        self.save_hyperparameters()
        # input layers
        self.input_embedding = nn.Sequential(nn.Embedding(
            n_tokens, model_dim), nn.Dropout(input_dropout))
        if E_position == "sinusoidal":
            # TODO: makes sure max_len also includes special tokens e.g. [CLS] and [SEP]
            self.positional_encoding = PositionalEncoding(
                model_dim, max_len=max_len)
        if E_position == "learned":
            self.positional_encoding = LearnedPositionalEncoding(
                model_dim, max_len=max_len)
        # multi-head transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            model_dim, T_n_head, T_hidden_dim, T_dropout, norm_first=T_norm_first, activation=nn_act, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, T_n_layers)
        # output layers for each sub-task in ["CLS","MLM","LA","NSP"]
        if "CLS" in training_objectives:
            assert n_classes is not None and n_classes > 1
            self.CLS_predictor = OutputLinears(
                model_dim, n_classes, output_dropout, nn_act)
        if "MLM" in training_objectives:
            if not token_pred_transposed:
                # TODO: remove unused target tokens (MLM is only done on regular tokens)
                self.MLM_predictor = OutputLinears(
                    model_dim, n_tokens, output_dropout, nn_act)
        if "LA" in training_objectives:
            if not token_pred_transposed:
                # TODO: remove unused target tokens (next token prediction is only done on regular tokens)
                self.LA_predictor = OutputLinears(
                    model_dim, n_tokens, output_dropout, nn_act)
                # TODO: cf GPT paper eqn. 2, replace output layer by transposed of token embedding matrix
        if "NSP" in training_objectives:
            self.NSP_predictor = OutputLinears(
                model_dim, 2, output_dropout, nn_act)
            # TODO: NSP task is done with encoding the input with additional segment embeddings
            self.NSP_embedding_A = nn.Parameter(torch.randn(model_dim))
            self.NSP_embedding_B = nn.Parameter(torch.randn(model_dim))
        # self.train_acc = torchmetrics.Accuracy()
        # self.valid_acc = torchmetrics.Accuracy()
        # self.test_acc = torchmetrics.Accuracy()
        # self.init_weights() # TODO: take care of specific init WITHOUT OVERWRITTING BUFFERS e.g. positional_encoding

    # def init_weights(self):
    #     raise NotImplementedError

    def configure_optimizers(self):
        if self.hparams.optim_model=="adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if self.hparams.optim_model=="adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def encode(self, x, mask=None, src_key_padding_mask=None, add_positional_encoding=True, add_segment_embedding=None):
        x = self.input_embedding(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        if add_segment_embedding is not None:
            x[torch.where(add_segment_embedding == 1)] = x[torch.where(
                add_segment_embedding == 1)]+self.NSP_embedding_A
            x[torch.where(add_segment_embedding == 0)] = x[torch.where(
                add_segment_embedding == 0)]+self.NSP_embedding_B
        # x = self.transformer_encoder(x, mask=mask)
        # TODO: check the use of src_key_padding_mask
        # expects src_mask: (S, S) and src_key_padding_mask: (N, S)
        # --> triangular lookahead mask is src_mask and column padding masks are src_key_padding_mask
        x = self.transformer_encoder(
            x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x
    
    # TODO: split into calculate_preds and then calculate_losses
    def calculate_losses(self, mb_dict):
        for _k in mb_dict:
            mb_dict[_k] = mb_dict[_k].to(self.device)
        # there are several forwards depending on the task (different masks and predictors)
        # CLS == inputs (padded_data,padding_mask) ; targets (cntry_labels)
        # MLM == inputs (masked_data,padding_mask/masking_mask) ; targets (padded_data[masking_pos])
        # LA == inputs (lookahead_inputs,lookahead_src_mask,lookahead_key_mask) ; targets (lookahead_targets[lookahead_pos])
        # NSP == inputs (nsp_data,nsp_mask,nsp_posA) ; targets (nsp_labels)
        # TODO: check correct reshape in MLM .view(-1,self.hparams.model_dim)[mb_dict["masking_pos"]] and LA .view(-1,self.hparams.model_dim)[mb_dict["lookahead_pos"]]
        # TODO: check cls_mode=="avg" only on outputs of not masked position
        loss_dict = dict()
        acc_dict = dict()
        if "CLS" in self.hparams.training_objectives:
            # pred_labels = self.encode(mb_dict["padded_data"],mask=mb_dict["padding_mask"])
            if self.hparams.cls_masked:
                # in BERT, all tasks are done on masked inputs
                # TODO: allow also NSP on masked inputs
                pred_labels = self.encode(
                    mb_dict["masked_data"], src_key_padding_mask=mb_dict["padding_mask"])
            else:
                pred_labels = self.encode(
                    mb_dict["padded_data"], src_key_padding_mask=mb_dict["padding_mask"])
            if self.hparams.detach_CLS:
                pred_labels = pred_labels.detach() # the supervised objective does not backpropagate to the encoder
            if self.hparams.cls_mode == "start":
                pred_labels = self.CLS_predictor(
                    pred_labels[:, 0, :])  # on the first [CLS] position
            if self.hparams.cls_mode == "avg":
                pred_labels = self.CLS_predictor(torch.mean(
                    pred_labels, 1))  # averaging over sequence dim
            mb_dict["pred_labels"] = pred_labels
            loss_dict["CLS_loss"] = F.cross_entropy(
                pred_labels, mb_dict["cntry_labels"])*self.hparams.lamb_CLS # weight factor on the supervised loss
            acc_dict["CLS_acc"] = (pred_labels.argmax(
                dim=-1) == mb_dict["cntry_labels"]).float().mean()
        if "MLM" in self.hparams.training_objectives:
            # pred_masked_tokens = self.encode(mb_dict["masked_data"],mask=mb_dict["padding_mask"]).view(-1,self.hparams.model_dim)[mb_dict["masking_pos"]] # check correct use of masking
            pred_masked_tokens = self.encode(mb_dict["masked_data"], src_key_padding_mask=mb_dict["padding_mask"]).view(
                -1, self.hparams.model_dim)[mb_dict["masking_pos"]]
            if not self.hparams.token_pred_transposed:
                pred_masked_tokens = self.MLM_predictor(pred_masked_tokens)
            else:
                pred_masked_tokens = torch.matmul(
                    pred_masked_tokens, self.input_embedding[0].weight.t())
            mb_dict["pred_masked_tokens"] = pred_masked_tokens
            loss_dict["MLM_loss"] = F.cross_entropy(
                pred_masked_tokens, mb_dict["padded_data"].view(-1)[mb_dict["masking_pos"]])
            acc_dict["MLM_acc"] = (pred_masked_tokens.argmax(
                dim=-1) == mb_dict["padded_data"].view(-1)[mb_dict["masking_pos"]]).float().mean()
        if "LA" in self.hparams.training_objectives:
            # pred_next_tokens = self.encode(mb_dict["lookahead_inputs"],mask=mb_dict["lookahead_mask"]).view(-1,self.hparams.model_dim)[mb_dict["lookahead_pos"]]
            pred_next_tokens = self.encode(mb_dict["lookahead_inputs"], mask=mb_dict["lookahead_src_mask"],
                                           src_key_padding_mask=mb_dict["lookahead_key_mask"]).view(-1, self.hparams.model_dim)[mb_dict["lookahead_pos"]]
            if not self.hparams.token_pred_transposed:
                pred_next_tokens = self.LA_predictor(pred_next_tokens)
            else:
                pred_next_tokens = torch.matmul(
                    pred_next_tokens, self.input_embedding[0].weight.t())
            mb_dict["pred_next_tokens"] = pred_next_tokens
            loss_dict["LA_loss"] = F.cross_entropy(
                pred_next_tokens, mb_dict["lookahead_targets"].reshape(-1)[mb_dict["lookahead_pos"]])
            acc_dict["LA_acc"] = (pred_next_tokens.argmax(
                dim=-1) == mb_dict["lookahead_targets"].reshape(-1)[mb_dict["lookahead_pos"]]).float().mean()
        if "NSP" in self.hparams.training_objectives:
            # pred_nsp = self.encode(mb_dict["nsp_data"],mask=mb_dict["nsp_mask"])
            pred_nsp = self.encode(
                mb_dict["nsp_data"], src_key_padding_mask=mb_dict["nsp_mask"], add_segment_embedding=mb_dict["nsp_posA"])
            if self.hparams.cls_mode == "start":
                # on the first [CLS] position
                pred_nsp = self.NSP_predictor(pred_nsp[:, 0, :])
            if self.hparams.cls_mode == "avg":
                # averaging over sequence dim
                pred_nsp = self.NSP_predictor(torch.mean(pred_nsp, 1))
            mb_dict["pred_nsp"] = pred_nsp
            loss_dict["NSP_loss"] = F.cross_entropy(
                pred_nsp, mb_dict["nsp_labels"])
            acc_dict["NSP_acc"] = (pred_nsp.argmax(
                dim=-1) == mb_dict["nsp_labels"]).float().mean()
        return loss_dict, mb_dict, acc_dict

    def gradient_check(self, mb_dict):
        # TODO: make a callback ?
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        loss_dict, _, _ = self.calculate_losses(mb_dict)
        losses = ["tot_loss"]+list(loss_dict.keys())
        for _l in losses:
            if _l == "tot_loss":
                loss = 0
                for _k in loss_dict:
                    loss = loss+loss_dict[_k]
            else:
                loss = loss_dict[_l]
            print("\n*** "+_l+" initial gradient check")
            optimizer.zero_grad()
            loss.backward()
            tot_grad = 0
            named_p = self.named_parameters()
            for name, param in named_p:
                if param.grad is not None:
                    sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                    if sum_abs_paramgrad == 0:
                        print(name, "sum_abs_paramgrad==0")
                    else:
                        tot_grad += sum_abs_paramgrad
                else:
                    print(name, "param.grad is None")
            print("tot_grad = ", tot_grad)
            loss_dict, _, _ = self.calculate_losses(mb_dict)
        optimizer.zero_grad()

    def log_metrics(self, mode, loss_dict, acc_dict, batch_size):
        tot_loss = 0
        for _k in loss_dict:
            tot_loss = tot_loss+loss_dict[_k]
            self.log(
                mode+"_"+_k, loss_dict[_k], on_step=False, on_epoch=True, batch_size=batch_size)
        for _k in acc_dict:
            self.log(
                mode+"_"+_k, acc_dict[_k], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(mode+"_tot_loss", tot_loss, on_step=False,
                 on_epoch=True, batch_size=batch_size)
        return tot_loss

    def training_step(self, mb_dict, batch_idx):
        batch_size = mb_dict["padded_data"].shape[0]
        loss_dict, mb_dict, acc_dict = self.calculate_losses(mb_dict)
        tot_loss = self.log_metrics("train", loss_dict, acc_dict, batch_size)
        return tot_loss

    def validation_step(self, mb_dict, batch_idx):
        batch_size = mb_dict["padded_data"].shape[0]
        loss_dict, mb_dict, acc_dict = self.calculate_losses(mb_dict)
        _ = self.log_metrics("valid", loss_dict, acc_dict, batch_size)

    # TODO: implement specific evaluations rather than logging losses ..
    # def test_step(self, mb_dict, batch_idx):
    #     batch_size = mb_dict["padded_data"].shape[0]
    #     loss_dict,mb_dict,acc_dict = self.calculate_losses(mb_dict)
    #     _ = self.log_metrics("test",loss_dict,acc_dict,batch_size)
    
    # TODO: double-check the functions below
    
    def compute_selfattention(self,x,mask,src_key_padding_mask,i_layer,d_model,num_heads):
        h = F.linear(x, self.transformer_encoder.layers[i_layer].self_attn.in_proj_weight,
                     bias=self.transformer_encoder.layers[i_layer].self_attn.in_proj_bias)
        qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
        d_k = q.size()[-1]
        attn_probs = attn_logits / math.sqrt(d_k)
        # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
        combined_mask = torch.zeros_like(attn_probs)
        if mask is not None:
            combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
        if src_key_padding_mask is not None:
            combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[1],1)
            # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
        combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
        # setting masked logits to -inf before softmax
        attn_probs += combined_mask
        attn_probs = F.softmax(attn_probs, dim=-1)
        return attn_logits,attn_probs
    
    def extract_selfattention_maps(self, x, mask=None, src_key_padding_mask=None, add_positional_encoding=True, add_segment_embedding=None):
        # input embeddings
        x = self.input_embedding(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        if add_segment_embedding is not None:
            x[torch.where(add_segment_embedding == 1)] = x[torch.where(
                add_segment_embedding == 1)]+self.NSP_embedding_A
            x[torch.where(add_segment_embedding == 0)] = x[torch.where(
                add_segment_embedding == 0)]+self.NSP_embedding_B
        # forward through transformer_encoder
        attn_logits_maps = []
        attn_probs_maps = []
        num_layers = self.transformer_encoder.num_layers
        d_model = self.transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        norm_first = self.transformer_encoder.layers[0].norm_first
        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = x.clone()
                if norm_first:
                    h = self.transformer_encoder.layers[i].norm1(h)
                # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
                attn_logits,attn_probs = self.compute_selfattention(h,mask,src_key_padding_mask,i,d_model,num_heads)
                attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
                attn_probs_maps.append(attn_probs)
                # forward of layer i
                x = self.transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
        return attn_logits_maps,attn_probs_maps

    # @torch.no_grad()
    # def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
    #     raise NotImplementedError


class LoggingTestLossCallback(Callback):
    def on_train_epoch_end(self, trainer, model):
        print("\nRunning test loss logging")
        model.eval()
        with torch.no_grad():
            for _, mb_dict in enumerate(model.test_dataloader):
                batch_size = mb_dict["padded_data"].shape[0]
                loss_dict, mb_dict, acc_dict = model.calculate_losses(mb_dict)
                _ = model.log_metrics("test", loss_dict, acc_dict, batch_size)
        model.train()


if __name__ == "__main__":
    from data_names_utils import prepare_datasets_names, custom_collate_fn

    data_path = "data_names"
    valid_size = 0.1
    test_size = 0.1

    # fixed_len_range = None
    fixed_len_range = [4, 11]

    training_objectives = ["CLS", "MLM", "LA", "NSP"]
    batch_size = 2  # NSP requires batch_size>1

    # training_objectives = ["CLS","MLM","LA"]
    # batch_size = 1
    
    lamb_CLS = 0.3
    detach_CLS = True

    token_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2,
                  '[MASK]': 3}  # special tokens used for training
    n_special_tokens = len(token_dict)
    mask_val = -1e9
    random_masking_prob = 0.15  # masked LM
    random_splitting_prob = 0.5  # ~ next sentence prediction

    train_dataset, valid_dataset, test_dataset, token_dict, label_dict, max_len = prepare_datasets_names(
        data_path, fixed_len_range=fixed_len_range, token_dict=token_dict, valid_size=valid_size, test_size=test_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                                  collate_fn=partial(custom_collate_fn, training_objectives=training_objectives, token_dict=token_dict, n_special_tokens=n_special_tokens, max_len=max_len,
                                                     mask_val=mask_val, random_masking_prob=random_masking_prob, random_splitting_prob=random_splitting_prob))  # issue with multiple workers ...
    for i, mb_dict in enumerate(train_dataloader):
        break

    model_dim = 16
    input_dropout = 0.1
    E_position = "sinusoidal"  # "sinusoidal" or "learned"
    nn_act = "gelu"  # "relu" or "gelu"
    T_n_head = 2
    T_hidden_dim = 64
    T_dropout = 0.1
    T_norm_first = True
    T_n_layers = 2
    output_dropout = 0.1
    cls_mode = "start"  # "start" or "avg"
    cls_masked = True
    token_pred_transposed = False
    n_classes = len(label_dict)
    model = TransformerPredictor(len(token_dict), model_dim, input_dropout, max_len+2, training_objectives, lamb_CLS, detach_CLS,
                                 E_position, nn_act, T_n_head, T_hidden_dim, T_dropout, T_norm_first, T_n_layers,
                                 output_dropout, cls_mode, cls_masked, token_pred_transposed, n_classes=n_classes, lr=3e-4, warmup=250, max_iters=100000)
    loss_dict, mb_dict, acc_dict = model.calculate_losses(mb_dict)
    model.gradient_check(mb_dict)
    
    attn_logits_maps,attn_probs_maps = model.extract_selfattention_maps(mb_dict["padded_data"], mask=None, src_key_padding_mask=mb_dict["padding_mask"], add_positional_encoding=True, add_segment_embedding=None)

    """
    # example configuration
    
    a_config = dict()
    a_config["valid_size"] = 0.1
    a_config["test_size"] = 0.1
    a_config["fixed_len_range"] = [4,11]
    a_config["training_objectives"] = ["CLS","MLM","LA","NSP"]
    a_config["lamb_CLS"] = 0.3
    a_config["detach_CLS"] = False
    a_config["token_dict"] = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    a_config["mask_val"] = -1e9
    a_config["random_masking_prob"] = 0.15
    a_config["random_splitting_prob"] = 0.5
    a_config["model_dim"] = 256
    a_config["input_dropout"] = 0.2
    a_config["E_position"] = "sinusoidal"
    a_config["nn_act"] = "gelu"
    a_config["T_n_head"] = 2
    a_config["T_hidden_dim"] = 256
    a_config["T_dropout"] = 0.2
    a_config["T_norm_first"] = True
    a_config["T_n_layers"] = 2
    a_config["output_dropout"] = 0.2
    a_config["cls_mode"] = "start"
    a_config["cls_masked"] = True
    a_config["token_pred_transposed"] = False
    with open(os.path.join("configs","a_config.json"), 'w') as f:
        json.dump(a_config, f, sort_keys=True, indent=4)
    """
