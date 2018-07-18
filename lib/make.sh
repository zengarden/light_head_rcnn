pushd ./utils/py_faster_rcnn_utils
make;
popd

pushd lib_kernel/lib_psroi_pooling
sh make.sh
popd

pushd lib_kernel/lib_roi_pooling
sh make.sh
popd

pushd lib_kernel/lib_roi_align
sh make.sh
popd

pushd lib_kernel/lib_psalign_pooling
sh make.sh
popd

pushd lib_kernel/lib_nms_dev
sh make.sh
popd

pushd datasets_odgt/lib_coco/PythonAPI
make install
popd
