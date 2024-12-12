#!/bin/bash

# 编译 CUDA 程序
nvcc -o add add.cu
nvcc -o sub sub.cu

# 每 2 秒钟执行一次
while true
do
    ./add
    ./sub
    ./sub
    sleep 1
done
