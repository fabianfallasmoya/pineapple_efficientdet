# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import os
import glob

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, CustomDataParallel, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def display(preds, imgs,savingPath,imageName,fileDetectionsYolo, imshow=True, imwrite=False,yoloFile=False):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                continue
    
            imgs[i] = imgs[i].copy()
    
            for j in range(len(preds[i]['rois'])):
                x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
                obj = obj_list[preds[i]['class_ids'][j]]
                score = float(preds[i]['scores'][j])
                plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])
                
                if yoloFile:
                    #####-Yolo's Format
                    xcenter = (abs(x1+x2)/2)/imgs[i].shape[1]
                    ycenter = (abs(y1+y2)/2)/imgs[i].shape[0]
                    bbox_width = (abs(x2-x1))/imgs[i].shape[1]
                    bbox_height = (abs(x2-x1))/imgs[i].shape[0]
                    fileDetectionsYolo.write(str(preds[i]['class_ids'][j]) + " " + 
                                             str(xcenter) + " " + str(ycenter) + " " + 
                                             str(bbox_width) + " " + str(bbox_height) + "\n")
    
    
            if imshow:
                cv2.imshow('img', imgs[i])
                cv2.waitKey(0)
    
            if imwrite:
                cv2.imwrite(f'{savingPath}/{imageName}_inferred.jpg', imgs[i])
            
            
    
def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join('test/', 'weightsInstance'))
    else:
        #torch.save(model.model.state_dict(), os.path.join('test/', name))
        torch.save(model.bifpn.state_dict(), os.path.join('test/', f'{name}-bifpn-weights.pth')) #saving bifpn weights
        torch.save(model.regressor.state_dict(), os.path.join('test/', f'{name}-regressor-weights.pth'))# saving regressor weights
        torch.save(model.classifier.state_dict(), os.path.join('test/', f'{name}-classifier-weights.pth'))# saving classifier weights
        torch.save(model.backbone_net.state_dict(), os.path.join('test/', f'{name}-backbone_net-weights.pth'))# saving backbone_net weights


dir_of_images_path = 'D:/Manfred/InvestigacionPinas/Beca-CENAT/workspace/multispectral-hiperspectral/Gira 10 13 Mar21/Lote 71/5_meters'
#dir_of_images_path = 'test/ms-dataset'
myImages = glob.glob(dir_of_images_path+'/*.JPG')
for img_path in myImages:
    imageName = img_path[len(dir_of_images_path)+1:len(img_path)-4]
    compound_coef = 4
    force_input_size = None  # set None to use default size
    #img_path = 'test/unknown.jpg'
    #img_path = 'test/unknown.jpg'
    
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    
    threshold = 0.4
    iou_threshold = 0.4
    
    use_cuda = False
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    
    '''
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']#'''
    obj_list = ['pineapple']
    
    
    color_list = standard_to_bgr(STANDARD_COLORS)
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    #model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth', map_location='cpu'))
    model.load_state_dict(torch.load(f'weights/efficientdet-d4_trained_weights.pth', map_location='cpu'))
    
    """
    model.bifpn.load_state_dict(torch.load(f'test/efficientdet-d4-trained-bifpn-weights.pth', map_location='cpu'))
    model.regressor.load_state_dict(torch.load(f'test/efficientdet-d4-trained-regressor-weights.pth', map_location='cpu'))
    model.classifier.load_state_dict(torch.load(f'test/efficientdet-d4-trained-classifier-weights.pth', map_location='cpu'))
    model.backbone_net.load_state_dict(torch.load(f'test/efficientdet-d4-trained-backbone_net-weights.pth', map_location='cpu'))
    """
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
        
        out, pyramid_count = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold,
                          model.pyramid_limits)
    
    
    
    
    out = invert_affine(framed_metas, out)
    fileDetectionsYolo = open(dir_of_images_path+'/'+imageName+'.txt', "w")
    display(out, ori_imgs,dir_of_images_path,imageName,fileDetectionsYolo, imshow=False, imwrite=False,yoloFile=True)
    #display(out, ori_imgs,dir_of_images_path,imageName,fileDetectionsYolo, imshow=False, imwrite=True,yoloFile=True)
    fileDetectionsYolo.close()
    #display(out, ori_imgs, imshow=False, imwrite=True)
    print(f"Pyramid count: {imageName}")
    #print(pyramid_count)
    #save_checkpoint(model,f'efficientdet-d{compound_coef}-trained')
print('done!')

'''
print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    # uncomment this if you want a extreme fps test
    # print('test2: model inferring only')
    # print('inferring images for batch_size 32 for 10 times...')
    # t1 = time.time()
    # x = torch.cat([x] * 32, 0)
    # for _ in range(10):
    #     _, regression, classification, anchors = model(x)
    #
    # t2 = time.time()
    # tact_time = (t2 - t1) / 10
    # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')#'''