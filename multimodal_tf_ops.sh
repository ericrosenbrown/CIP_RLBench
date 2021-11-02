#!/bin/bash

# pip install -r reqs_multimodal.txt

## If one's python or cuda environment is different then change the following variables
ENV_PATH=/home/abba/envs/cip3.6/
TF_INC="$ENV_PATH/lib/python3.6/site-packages/tensorflow/include"
TF_LIB="$ENV_PATH/lib/python3.6/site-packages/tensorflow"
CUDA_PATH="/usr/local/cuda"
# CUDA_PATH=$CUDA_ROOT_DIR # on ccv, replace above
NVCC_VERSION="$CUDA_PATH/bin/nvcc"
PATH_TO_TF_OPS=$HOME/CIP_RLBench/MultiModalGrasping/scripts/PointNet2/tf_ops

## fix missing .so
cd $ENV_PATH/lib/python3.6/site-packages/tensorflow
cp libtensorflow_framework.so.1 libtensorflow_framework.so

## compile tf_ops
cd $PATH_TO_TF_OPS
echo "Installing PointNet's various tf operations"
echo "-----------------------"
OPERATION="sampling"
echo $OPERATION
cd sampling
$NVCC_VERSION tf_${OPERATION}_g.cu -o tf_${OPERATION}_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_${OPERATION}.cpp tf_${OPERATION}_g.cu.o -o tf_${OPERATION}_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

echo "-----------------------"
OPERATION="grouping"
echo $OPERATION
cd grouping
$NVCC_VERSION tf_${OPERATION}_g.cu -o tf_${OPERATION}_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_${OPERATION}.cpp tf_${OPERATION}_g.cu.o -o tf_${OPERATION}_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

echo "-----------------------"
OPERATION="3d_interpolation"
echo $OPERATION
cd $OPERATION
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_INC -I $CUDA_PATH/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
cd ..

echo "-----------------------"
echo "DONE"
echo "-----------------------"
