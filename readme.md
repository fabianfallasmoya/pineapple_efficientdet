## EfficientDet Pytorch

The pytorch re-implement of the official [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) with SOTA performance in real time, original paper link: <https://arxiv.org/abs/1911.09070>. This repository is based on the PyTorch implementation from: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## Performance

## Pretrained weights and benchmark

The performance is very close to the paper's, it is still SOTA.

The speed/FPS test includes the time of post-processing with no jit/data precision trick.

| coefficient | pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 33.1 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.8 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 42.1 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 45.6 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.8 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 50.2 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.7 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth) | 3819 | 3.73 | - | 52.7 | 53.7
| D7X | [efficientdet-d8.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth) | 3983 | 2.39 | - | 53.9 | 55.1

## Installation

    # install requirements
    pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
    pip install torch==1.4.0
    pip install torchvision==0.5.0
     
    # run the simple inference script
    python efficientdet_test.py
 
## Training

### 1. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    
    # for example, coco2017
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val2017/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train2017.json
                -instances_val2017.json

### 2. Manual set project's specific parameters

    # create a yml file {your_project_name}.yml under 'projects'folder 
    # modify it following 'coco.yml'
     
    # for example
    project_name: coco
    train_set: train
    val_set: val
    num_gpus: 1  # 0 means using cpu, 1-N means using gpus 
    
    # mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
    # this is coco anchors, change it if necessary
    anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    
    # objects from all labels from your dataset with the order from your annotations.
    # its index must match your dataset's category_id.
    # category_id is one_indexed,
    # for example, index of 'car' here is 2, while category_id of is 3
    obj_list: ['person', 'bicycle', 'car', ...]

### 3.a. Train on coco from scratch(not necessary)

    # train efficientdet-d0 on coco from scratch 
    # with batchsize 12
    # This takes time and requires change 
    # of hyperparameters every few hours.
    # If you have months to kill, do it. 
    # It's not like someone going to achieve
    # better score than the one in the paper.
    # The first few epoches will be rather unstable,
    # it's quite normal when you train from scratch.
    
    python train.py -c 0 --batch_size 64 --optim sgd --lr 8e-2

### 3.b. Train a custom dataset from scratch

    # train efficientdet-d1 on a custom dataset 
    # with batchsize 8 and learning rate 1e-5
    
    python train.py -c 1 -p your_project_name --batch_size 8 --lr 1e-5

### 3.c. Train a custom dataset with pretrained weights (Highly Recommended)

    # train efficientdet-d2 on a custom dataset with pretrained weights
    # with batchsize 8 and learning rate 1e-3 for 10 epoches
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth
    
    # with a coco-pretrained, you can even freeze the backbone and train heads only
    # to speed up training and help convergence.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --num_epochs 10 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True

### 4. Early stopping a training session

    # while training, press Ctrl+c, the program will catch KeyboardInterrupt
    # and stop training, save current checkpoint.

### 5. Resume training

    # let says you started a training session like this.
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights /path/to/your/weights/efficientdet-d2.pth \
     --head_only True
     
    # then you stopped it with a Ctrl+c, it exited with a checkpoint
    
    # now you want to resume training from the last checkpoint
    # simply set load_weights to 'last'
    
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 \
     --load_weights last \
     --head_only True

### 6. Evaluate model performance

    # eval on your_project, efficientdet-d5
    
    python coco_eval.py -p your_project_name -c 5 \
     -w /path/to/your/weights

### 7. Debug training (optional)

    # when you get bad result, you need to debug the training result.
    python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-3 --debug True
    
    # then checkout test/ folder, there you can visualize the predicted boxes during training
    # don't panic if you see countless of error boxes, it happens when the training is at early stage.
    # But if you still can't see a normal box after several epoches, not even one in all image,
    # then it's possible that either the anchors config is inappropriate or the ground truth is corrupted.

## TODO

- [X] re-implement metrics code from Padilla.
- [X] adapt anchor strategies
- [X] training-scripts
- [X] improve comments
