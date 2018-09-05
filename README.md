# PCL: Proposal Cluster Learning for Weakly Supervised Object Detection 

By [Peng Tang](https://pengtang.xyz/), [Xinggang Wang](http://www.xinggangw.info/), [Song Bai](http://songbai.site/), [Wei Shen](http://songbai.site/), [Xiang Bai](http://122.205.5.5:8071/~xbai/), [Wenyu Liu](http://mclab.eic.hust.edu.cn/MCWebDisplay/PersonDetails.aspx?Name=Wenyu%20Liu), and [Alan Yuille](http://www.cs.jhu.edu/~ayuille/).

**The codes to train and eval our original OICR using PyTorch as backend is available [here](https://github.com/vadimkantorov/caffemodel2pytorch/blob/master/README.md).
Thanks [Vadim](http://vadimkantorov.com/)!**

**The [original implementation](https://github.com/ppengtang/oicr/tree/pcl) is based on the caffe which only supports single-gpu training for python.**

### Introduction

**Proposal Cluster Learning (PCL)** is a framework for weakly supervised object detection with deep ConvNets. 
 - It achieves state-of-the-art performance on weakly supervised object detection (Pascal VOC 2007 and 2012, ImageNet DET).
 - Our code is written by C++ and Python, based on [Caffe](http://caffe.berkeleyvision.org/), [fast r-cnn](https://github.com/rbgirshick/fast-rcnn), [faster r-cnn](https://github.com/rbgirshick/py-faster-rcnn), and [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU).

The original paper has been accepted by CVPR 2017. This is an extened version.
For more details, please refer to [here](https://arxiv.org/abs/1704.00138) and [here](https://arxiv.org/abs/1807.03342).

### Comparison with other methods
(a) Conventional MIL method;
(b) Our original OICR method with newly proposed proposal cluster generation method;
(c) Our PCL method.

<p align="left">
<img src="images/method_compare.jpg" alt="method compare" width="500px">

### Architecture

<p align="left">
<img src="images/architecture.jpg" alt="PCL architecture" width="900px">
</p>

### Results

| Method | VOC2007 test *mAP* | VOC2007 trainval *CorLoc* | VOC2012 test *mAP* | VOC2012 trainval *CorLoc* | ImageNet *mAP*
|:-------|:-----:|:-------:|:-------:|:-------:|:-------:|
| PCL-VGG_M | 40.8 | 59.6 | 37.6 | 62.9 | 14.4 |
| PCL-VGG16 | 43.5 | 62.7 | 40.6 | 63.2 | 18.4 |
| PCL-Ens. | 45.8 | 63.0 | 41.6 | 65.0 | 18.8 |
| PCL-Ens.+FRCNN | 48.8 | 66.6 | 44.2 | 68.0 | 19.6 |

### Visualizations

Some PCL visualization results.
<p align="left">
<img src="images/detections.jpg" alt="Some visualization results" width="900px">
</p>

Some visualization comparisons among WSDDN, WSDDN+context, and PCL.
<p align="left">
<img src="images/detections_compare.jpg" alt="Some visualization comparisons among WSDDN, WSDDN+context, and PCL" width="900px">
</p>

### License

PCL is released under the MIT License (refer to the LICENSE file for details).

### Citing PCL

If you find PCL useful in your research, please consider citing:

    @article{tang2018pcl,
        author = {Tang, Peng and Wang, Xinggang and Bai, Song and Shen, Wei and Bai, Xiang and Liu, Wenyu and Yuille, Alan},
        title = {{PCL}: Proposal Cluster Learning for Weakly Supervised Object Detection},
        journal = {arXiv preprint arXiv:1807.03342},
        year = {2018}
    }

    @inproceedings{tang2017multiple,
        author = {Tang, Peng and Wang, Xinggang and Bai, Xiang and Liu, Wenyu},
        title = {Multiple Instance Detection Network with Online Instance Classifier Refinement},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        pages = {3059--3067},
        year = {2017}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation)
4. [Installation for training and testing](#installation-for-training-and-testing)
5. [Extra Downloads (selective search)](#download-pre-computed-selective-search-object-proposals)
6. [Extra Downloads (ImageNet models)](#download-pre-trained-imagenet-models)
7. [Extra Downloads (Models trained on PASCAL VOC)](#download-models-trained-on-pascal-voc)
8. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  ```

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`, `sklearn`
3. MATLAB

### Requirements: hardware

1. NVIDIA GTX TITANX (~12G of memory)

### Installation

1. Clone the PCL repository
  ```Shell
  git clone https://github.com/ppengtang/oicr.git & cd oicr
  git checkout pcl
  git clone https://github.com/ppengtang/caffe.git
  ```

2. Build the Cython modules
  ```Shell
  cd $PCL_ROOT/lib
  make
  ```
    
3. Build Caffe and pycaffe
  ```Shell
  cd $PCL_ROOT/caffe
  # Now follow the Caffe installation instructions here:
  #   http://caffe.berkeleyvision.org/installation.html

  # If you're experienced with Caffe and have all of the requirements installed
  # and your Makefile.config in place, then simply do:
  make all -j 8
  make pycaffe
  ```

### Installation for training and testing
1. Download the training, validation, test data and VOCdevkit

  ```Shell
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
  ```
2. Extract all of these tars into one directory named `VOCdevkit`

  ```Shell
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_18-May-2011.tar
  ```
3. It should have this basic structure

  ```Shell
  $VOCdevkit/                           # development kit
  $VOCdevkit/VOCcode/                   # VOC utility code
  $VOCdevkit/VOC2007                    # image sets, annotations, etc.
  # ... and several other directories ...
  ```

4. Create symlinks for the PASCAL VOC dataset

  ```Shell
  cd $PCL_ROOT/data
  ln -s $VOCdevkit VOCdevkit2007
  ```
  Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.

5. [Optional] follow similar steps to get PASCAL VOC 2012.

6. You should put the generated proposal data under the folder $PCL_ROOT/data/selective_search_data, with the name "voc_2007_trainval.mat", "voc_2007_test.mat", just as the form of [fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

7. The pre-trained models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). You should put it under the folder $PCL_ROOT/data/imagenet_models, just as the form of [fast-rcnn](https://github.com/rbgirshick/fast-rcnn).

### Download pre-computed Selective Search object proposals

Pre-computed selective search boxes can also be downloaded for VOC2007 and VOC2012.

  ```Shell
  cd $PCL_ROOT
  ./data/scripts/fetch_selective_search_data.sh
  ```

This will populate the `$PCL_ROOT/data` folder with `selective_selective_data`.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded.

  ```Shell
  cd $PCL_ROOT
  ./data/scripts/fetch_imagenet_models.sh
  ```
These models are all available in the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but are provided here for your convenience.
(The script is copied from the [fast-rcnn](https://github.com/rbgirshick/fast-rcnn)).

### Download models trained on PASCAL VOC

Models trained on PASCAL VOC can be downloaded [here](https://drive.google.com/drive/folders/1aqhAxPgrHdJoMncddQ73YiNXtiENEX3b?usp=sharing).

### Usage

**Train** a PCL network. For example, train a VGG16 network on VOC 2007 trainval

  ```Shell
  ./tools/train_net_multi_gpu.py --gpu_id 0,1 --solver models/VGG16/solver_voc2007.prototxt \
    --weights data/imagenet_models/$VGG16_model_name --iters 50000 --imdb voc_2007_trainval
  ```

**Test** a PCL network. For example, test the VGG 16 network on VOC 2007 test:

#### On trainval
  ```Shell
  ./tools/test_net.py --gpu 1 --def models/VGG16/test.prototxt \
    --net output/default/voc_2007_trainval/vgg16_pcl_iter_50000.caffemodel \
    --imdb voc_2007_trainval
  ```

#### On test
  ```Shell
  ./tools/test_net.py --gpu 1 --def models/VGG16/test.prototxt \
    --net output/default/voc_2007_trainval/vgg16_pcl_iter_50000.caffemodel \
    --imdb voc_2007_test
  ```

Test output is written underneath `$PCL_ROOT/output`.

#### Evaluation
For mAP, run the python code tools/reval.py
  ```Shell
  ./tools/reval.py $output_dir --imdb voc_2007_test --matlab
  ```

For CorLoc, run the python code tools/reval_discovery.py
  ```Shell
  ./tools/reval_discovery.py $output_dir --imdb voc_2007_trainval
  ```


The codes for training fast rcnn by pseudo ground truths are available on [here](https://github.com/ppengtang/fast-rcnn).
