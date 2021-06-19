import torch
import os
from torcjson_s import cudnn
import psutil

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt

import numpy as np
import yaml
import json

from PIL import Image
import PIL

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, preprocess_all, invert_affine, postprocess_original
from utils.semi_utils import get_image_json, insert_bbox

#LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(cpu_list):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])



if(True):
    #control print
    print('1 - setting up parameters.')

    #important parameters 
    iteration = 1
    project_name = 'apple_semi_annotated'
    throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39]) 
    compound_coef = 4
    force_input_size = None  # set None to use default size
    threshold = 0.4
    iou_threshold = 0.4
    label_confidence = 0.95
    obj_list = ['apple']
    ratios_ = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    scales_ = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    weights_ = 'logs/apple_semi_annotated/efficientdet-d4_trained_weights.pth'
    unlabeled_images_set = 'test'
    labeled_images_set = 'train'
    
    #Get images
    root_dir_images = f'datasets/{project_name}/{unlabeled_images_set}'
    original_names = os.listdir(root_dir_images)
    img_path = [root_dir_images + '/' + i for i in original_names]

    destination_dir_images_annotated = f'datasets/{project_name}/{labeled_images_set}_{str(iteration)}/'
    destination_labeled = f'datasets/{project_name}/{labeled_images_set}/'
    if not os.path.exists(destination_dir_images_annotated):
        os.mkdir(destination_dir_images_annotated)

    
    #Get original JSON
    root_dir_orig_json = f'datasets/{project_name}/annotations/instances_{labeled_images_set}.json'
    new_json_path = f'datasets/{project_name}/annotations/instances_{labeled_images_set}_{str(iteration)}.json'
    
    
    #Model variables
    use_cuda = False
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    
    #control print
    print('2 - Get predictions.')

    #GET predictions -> prepare the model
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess_all(img_path, max_size=input_size)
    
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
    
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    
    model = EfficientDetBackbone(compound_coef=compound_coef, 
                                 num_classes=len(obj_list),
                                 ratios=ratios_,
                                 scales=scales_)
    
    model.load_state_dict(torch.load(weights_))
    model.requires_grad_(False)
    model.eval()
    
    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()
    
    #GET predictions -> run predictions
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
    
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
    
        out = postprocess_original(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
    out = invert_affine(framed_metas, out)
    

    #Save images and new ground truth    
    #Get json to insert new images and new values
    with open(root_dir_orig_json, "r") as read_file:
        orig_json = json.load(read_file)
        new_json = orig_json
        update_json_file = False

        #control print
        print('3 - Keep prediction only above a threshold.')

        #Iter over ALL unlabeled images
        for i in range(len(ori_imgs)):
            #do nothing if there are no rois
            if len(out[i]['rois']) == 0:
                continue
            
            #Get a temporal JSON with the current image already inserted
            temporal_json, image_dict = get_image_json(new_json, original_names[i])
            update_current = False

            for j in range(len(out[i]['rois'])):

                #Include new values as Ground Truth
                if out[i]['scores'][j] >= label_confidence:
                    #Get coordinates
                    (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)

                    #Original code to save image
                    #---------------------------
                    cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                    obj = obj_list[out[i]['class_ids'][j]]
                    score = float(out[i]['scores'][j])
            
                    cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                    #---------------------------

                    #flag to indicate that we need to update the json with new ground truth
                    update_current = True
                    update_json_file = True
                    
                    #when loading categories this line is executed --> annotation[0, 4] = a['category_id'] - 1
                    #So, to save categories we have to do
                    category = int(out[i]['class_ids'][j] + 1)

                    #caculate width and height
                    width = x2 - x1
                    height = y2 - y1

                    #add bbox
                    temporal_json = insert_bbox(temporal_json, [float(x1), float(y1), float(width), float(height)], image_dict["id"], category)
            
            if update_current:
                new_json = temporal_json 
            
                #save image with all detected bboxes
                image_ = Image.fromarray(np.uint8(ori_imgs[i]), 'RGB')
                image_.save(destination_dir_images_annotated + original_names[i])

                #copy image to labeled images
                current_img = cv2.imread(root_dir_images + '/' + original_names[i])
                cv2.imwrite(destination_labeled + original_names[i], current_img) 
            #break

    #control print
    print('4 - Save new JSON.')
    #save new json file
    if update_json_file:
        with open(new_json_path, 'w') as json_file:
            json.dump(new_json, json_file)

    print('5 - Done.')
