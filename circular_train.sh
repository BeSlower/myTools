#!/usr/bin/env bash

cd $(dirname ${BASH_SOURCE[0]})/../

caffe=caffe/build/tools/
pretrain_model=model/googlenet/googlenet.caffemodel
solver=model/googlenet/solver.prototxt
result=result

model=model/googlenet/deploy.prototxt
blob=fc_embedding
feature_dir=feature/
num_iters=354  # num_img 42468 bath_size 120 num_iters 354 
               # num_img 42468 bath_size 36 num_iters 1180 
num_samples=42468

code=code/

max_epoch=20

for((idx=1;idx<=${max_epoch};idx++))
do
    i=$((10*idx))
    log=log/epoch_${i}.log

    # training
    GLOG_logtostderr=1 ${caffe}/caffe_googlenet train \
        -solver ${solver} -weights ${pretrain_model} -gpu 6,7 2>&1 | tee ${log}
    
    mv ${result}/*.caffemodel ${result}/epoch_${i}.caffemodel
    cp ${result}/epoch_${i}.caffemodel ${result}_backup/
    rm ${result}/*

    # update training setting
    pretrain_model=${result}_backup/epoch_${i}.caffemodel

    # extracting features
    trained_model=${pretrain_model}
    if [ -d ${feature_dir}/train_features_lmdb ]; then
        rm -rf ${feature_dir}/*
    fi
    ${caffe}/extract_features \
           ${trained_model} ${model} ${blob} \
           ${feature_dir}/train_features_lmdb \
           ${num_iters} lmdb GPU 6

    python ${code}/convert_lmdb_to_numpy.py \
        ${feature_dir}/train_features_lmdb ${feature_dir} \
        --truncate ${num_samples}
    
    # update training list
    mv feature.mat ${code}/
    mv train_list.txt train_list_${i}.txt
    mv train_list_${i}.txt ${result}_backup/train_list/
    matlab -nojvm < ${code}/genList4Triplet.m
    
    # update solver
    if(((idx%5)==0)); then
        cp model/googlenet/solver_set/solver_${idx}.prototxt model/googlenet/solver.prototxt
    fi
done