import torch
import os
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt

import numpy as np
import yaml
import json

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, preprocess_all, invert_affine, postprocess

#version para calcular visualize sobre todas las imagenes de un folder
if(True):
    compound_coef = 4
    force_input_size = None  # set None to use default size
    
    root_dir_testing = 'datasets/apple_semi_annotated/valid'
    destination_dir_images = root_dir_testing + '_results/'
    original_names = os.listdir(root_dir_testing)
    img_path = [root_dir_testing + '/' + i for i in original_names]
    print(img_path)
    
    threshold = 0.4
    iou_threshold = 0.4
    
    use_cuda = False
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    
    obj_list = ['apple']
    
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess_all(img_path, max_size=input_size)
    
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
    
                                 # replace this part with your project's anchor config
                                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    
    model.load_state_dict(torch.load('logs/apple_semi_annotated/efficientdet-d4_trained_weights.pth'))
    model.requires_grad_(False)
    model.eval()
    
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
    
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
    
        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
    
    out = invert_affine(framed_metas, out)
    
    from PIL import Image
    import PIL
        
    for i in range(len(ori_imgs)):
        print(len(ori_imgs))
        if len(out[i]['rois']) == 0:
            continue
    
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
    
            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            
            #print(type(ori_imgs[i]))
            #print(ori_imgs[i].shape)
            #print(ori_imgs[i][0])
            image_ = Image.fromarray(np.uint8(ori_imgs[i]), 'RGB')
        image_.save(destination_dir_images + original_names[i])