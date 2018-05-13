# Light-head R-CNN


## Introduction
We release code for [Light-Head R-CNN](https://arxiv.org/abs/1711.07264). 



This is my best practice for my research. 

This repo is organized as follows:

```
light_head_rcnn/
    |->experiments
    |    |->user
    |    |    |->your_models
    |->lib       
    |->tools
    |->output
```

## Main Results
1. We train on COCO trainval which includes `80k` training and `35k` validation images. Test on minival which is a `5k` subset in validation datasets. Noticing test-dev should be little higher than minival.
2. We provide some crutial ablation experiments details, and it is easy to diff the difference.
3. We share our training logs in [GoogleDrive](https://drive.google.com/open?id=1-Mqj385d_1t4wcmhl25TZO1g-uw5X-xK) output folder, which contains dump models, training loss and speed of each steps. (experiments are done on 8 titan xp, and 2batches/per_gpu. Training should be within one day.)
4. Because the limitation of the time, extra experiments are comming soon.

|         Model Name                                               |<sub>mAP@all</sub>|<sub>mAP@0.5</sub>|<sub>mAP@0.75</sub>|<sub>mAP@S</sub>|<sub>mAP@M</sub>|<sub>mAP@L</sub>| 
|-------------------------------------------------------------     |------------------|------------------|---------          |-------         |-------         |-------         |   
|<sub>R-FCN, ResNet-v1-101 </br> our reproduce baseline</sub>      | 35.5             | 54.3             |   33.8            | 12.8           | 34.9           | 46.1           |   
|<sub>Light-Head R-CNN </br> ResNet-v1-101</sub>                   | 38.2             | 60.9             |   41.0            | 20.9           | 42.2           | 52.8           |   
|<sub>Light-Head,ResNet-v1-101 </br> +align pooling </sub>         | 39.3             | 61.0             |   42.4            | 22.2           | 43.8           | 53.2           |   
|<sub>Light-Head,ResNet-v1-101 </br> +align pooling  + nms0.5</sub>| 40.0             | 62.1             |   42.9            | 22.5           | 44.6           | 54.0           |  

Experiments path related to model:

```
experiments/lizeming/rfcn_reproduce.ori_res101.coco.baseline
experiments/lizeming/light_head_rcnn.ori_res101.coco 
experiments/lizeming/light_head_rcnn.ori_res101.coco.ps_roialign
experiments/lizeming/light_head_rcnn.ori_res101.coco.ps_roialign
```

## Requirements
1. tensorflow-gpu==1.5.0  (We only test on tensorflow 1.5.0, early tensorflow is not supported because of our gpu nms implementation)
2. python3. We recommend using Anaconda as it already includes many common packages. (python2 is not tested)
3. Python packages might missing. pls fix it according to the error message.

## Installation, Prepare data, Testing, Training
### Installation
1. Clone the Light-Head R-CNN repository, and we'll call the directory that you cloned Light-Head R-CNNN as `${lighthead_ROOT}`.

```
git clone https://github.com/zengarden/light_head_rcnn
``` 

2. Compiling

```
cd ${lighthead_ROOT}/lib;
bash make.sh
``` 

Make sure all of your compiling is successful. It may arise some errors, it is useful to find some common compile errors in [FAQ](##FAQ)

3. Create log dump directory, data directory. 

```
cd ${lighthead_ROOT};
mkdir output
mkdir data
``` 

### Prepare data
data should be organized as follows:

```
data/
    |->imagenet_weights/res101.ckpt
    |->MSCOCO
    |    |->odformat
    |    |->instances_xxx.json
    |    |train2014
    |    |val2014
```
Download res101 basemodel:

```
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
``` 

We transfer instances_xxx.json to odformat(object detection format), each line in odformat is an annotation(json) for one image. Our transformed odformat is  shared in [GoogleDrive](https://drive.google.com/open?id=1-Mqj385d_1t4wcmhl25TZO1g-uw5X-xK) odformat.zip .
### Testing

1. Using `-d` to assign gpu_id for testing. (e.g.  `-d 0,1,2,3`   or `-d 0-3` )
2. Using `-s` to visualize the results. 
3. Using '-se' to specify start_epoch for testing.

We share our experiments output(logs) folder in [GoogleDrive](https://drive.google.com/open?id=1-Mqj385d_1t4wcmhl25TZO1g-uw5X-xK). Download it and place it to `${lighthead_ROOT}`, then test our release model.

e.g.

```
cd experiments/lizeming/light_head_rcnn.ori_res101.coco.ps_roialign
python3 test.py -d 0-7 -se 26
``` 

### Training

We provide common used train.py in tools, which can be linked to experiments folder.

e.g.
```
cd experiments/lizeming/light_head_rcnn.ori_res101.coco.ps_roialign
python3 config.py -tool
cp tools/train.py .
python3 train.py -d 0-7
``` 

## Features 
 
This repo is designed be `fast` and `simple` for research. Such as: we build fast dataprovider for training, rewrite fast_nms. However there are still some can be improved: anchor_target and proposal_target layer are `tf.py_func`, which means it will run on cpu. 

## Disclaimer
This is an implementation for [Light-Head R-CNN](https://arxiv.org/abs/1711.07264), it is worth noting that:

* The original implementation is based on our internal Platform used in Megvii. There are slight differences in the final accuracy and running time due to the plenty details in platform switch.
* The code is tested on a server with 8 Pascal Titian XP gpu, 188.00 GB memory, and 40 core cpu.


## Citing Light-Head R-CNN

If you find Light-Head R-CNN is useful in your research, pls consider citing:

```
@article{li2017light,
  title={Light-Head R-CNN: In Defense of Two-Stage Object Detector},
  author={Li, Zeming and Peng, Chao and Yu, Gang and Zhang, Xiangyu and Deng, Yangdong and Sun, Jian},
  journal={arXiv preprint arXiv:1711.07264},
  year={2017}
}
```

## FAQ

* fatal error: cuda/cuda_config.h: No such file or directory

First, find where is cuda_config.h.

e.g.

```
find /usr/local/lib/ | grep cuda_config.h
```

then export your cpath, like:

```
export CPATH=$CPATH:/usr/local/lib/python3.5/dist-packages/external/local_config_cuda/cuda/
```
