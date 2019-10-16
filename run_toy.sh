#!/usr/bin/env bash

############################### 
# Training a mt-dnn models
# Note that this is a toy setting and please refer our paper for detailed hyper-parameters.
############################### 

python src/mt_dnn/embedding_utils/prepro_std.py
python src/mt_dnn/embedding_utils/train.py