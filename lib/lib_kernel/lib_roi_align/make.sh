TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
CUDA_PATH=/usr/local/cuda/

nvcc -std=c++11 -c -o roi_align_op.cu.o roi_align_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52 -I$TF_INC/external/nsync/public

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_align.so roi_align_op.cc \
	roi_align_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -L$TF_LIB -ltensorflow_framework  -I$TF_INC/external/nsync/public

# for gcc5-built tf
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=1 -o roi_align.so roi_align_op.cc \
#	roi_align_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
