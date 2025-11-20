#!/bin/bash -l


# -- Define hyperparameters --

hidden_dim=512
hidden_depth=1
dropout=0.0
batch_size=256
learning_rate=0.0002
weight_decay=0.0
num_epochs=10

# -- Execute job --

python3 train.py \
\
--mode patch-segmentation \
\
--split-path /path/to/split.csv \
--metadata-dirs /path/to/metadata \
--patches-dirs /path/to/patches \
--features-dirs /path/to/features \
--results-dir /path/to/results/segmentation/models \
\
--train-subsets 1 2 3 4 \
--val-subsets 0 \
--test-subsets 0 1 2 3 4 \
--targets 1 \
--annotation-thresholds 0.0 0.5 \
--drop-duplicates sample \
--sampler hierarchical_label \
\
--model-type mlp \
--input-dim 1024 \
--hidden-dim ${hidden_dim} \
--num-classes 2 \
--hidden-depth ${hidden_depth} \
--dropout ${dropout} \
\
--train-batch-size ${batch_size} \
--val-batch-size ${batch_size} \
--learning-rate ${learning_rate} \
--weight-decay ${weight_decay} \
--objective cross-entropy \
--num-epochs ${num_epochs} \
--val-interval 1 \
\
--test-checkpoint last \
\
--device cpu
