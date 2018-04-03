TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_PATH=/usr/local/cuda/

nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
        -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52 -I$TF_INC/external/nsync/public --expt-relaxed-constexpr

g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc \
    psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework  -I$TF_INC/external/nsync/public

# g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
#         psroi_pooling_op.cu.o -I $TF_INC  -D GOOGLE_CUDA=1 -fPIC -lcudart -L $CUDA_PATH/lib64
