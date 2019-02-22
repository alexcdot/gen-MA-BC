#!/bin/bash

python train.py \
--trial 112 \
--model MACRO_VRNN \
--x_dim 2 \
--y_dim 8 \
--z_dim 16 \
--h_dim 200 \
--m_dim 90 \
--rnn_dim 200 \
--rnn_micro_dim 200 \
--rnn_macro_dim 200 \
--n_trained_def 1 \
--n_trained_off 1 \
--n_gt_def 1 \
--n_gt_off 1 \
--n_layers 2 \
--subsample 1 \
--n_epochs 200 \
--clip 10 \
--start_lr 1e-4 \
--min_lr 1e-4 \
--batch_size 512 \
--cuda \
--pretrain 50
