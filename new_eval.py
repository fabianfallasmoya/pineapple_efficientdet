# -*- coding: utf-8 -*-

"""### Prepare the required repositories 
* https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

* https://github.com/rafaelpadilla/Object-Detection-Metrics

# Executing the Object Detection Metrics, through Rafael's Padilla code

### The next cell will be the only one to be changed
"""

import os
import sys
import shutil
import cv2
import torch
from torch.backends import cudnn
import argparse
import psutil

from backbone import EfficientDetBackbone
import matplotlib.pyplot as plt

import numpy as np
import yaml
import json
import pandas as pd
import csv 

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string, postprocess_pyramid_filter
from pyramid_analysis import PyramidAnalysis


#------------------------------------------------------------
###Parameters from the user
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, required=True, default=None, help='project file that contains parameters')
ap.add_argument('-pw', '--project_weights', type=str, required=False, default=None, help='project from the weights were trained')
ap.add_argument('-w', '--weights', type=str, required=True, default=None, help='name of the weights file') ## This file shuld be in EfficientDet/logs directory
ap.add_argument('--nms_threshold', type=float, default=0.4, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('-c', '--compound_coef', type=int, required=True, default=0, help='coefficients of efficientdet')
ap.add_argument('--confidence_threshold', type=float, default=0.0, help='confidence threshold used in the prediction of bounding boxes when is a single test')
ap.add_argument('--single_test', type=boolean_string, required=True, default=False,
                        help='true to set up a single evaluation with a specific confidence threshold or false for several tests in a range of 0.05 to 0.95 confidence threshold')
args = ap.parse_args()
project = args.project #This is the yml project name, also is the class name of the bboxes
project_weights = args.project_weights
wieghtsFileName = args.weights #The name of the weights file
nms_threshold = args.nms_threshold
coefficient = args.compound_coef ## EfficientDet coefficient
confidence_threshold = args.confidence_threshold

rootDir = os.getcwd() + '/' ##The directory where is going to be located the EfficientDet repo and Object Detection Metrics repo
#This directory is where the results of the evaluation will be stored
dirForGroundTruthAndDetections = rootDir + 'eval'
#############################################################
#------------------------------------------------------------


def get_batch(data, n=1):
    l = len(data)
    for ndx in range(0, l, n):
        yield data[ndx:min(ndx + n, l)]


def get_pyramid_mask(dir_images, weights_location):
    #read config file to access path to ground truth
    #params = yaml.safe_load(open(f'{rootDir}projects/{project_weights}'+'.yml'))
    #data_folder = params['train_set']

    im_save = False
    pyramid = PyramidAnalysis(weights_location, dir_images, im_save)
    
    
    original_names = os.listdir(dir_images)
    img_path = [dir_images+'/'+i for i in original_names]
    
    pyramid_acc = torch.zeros(5, dtype=torch.int32)
    img_id = 0
    for my_batch in get_batch(img_path, 3):
        pyramid_temp, img_id = pyramid.results_batch(my_batch, img_id)
        pyramid_acc += pyramid_temp
    
    
    return pyramid_acc


"""## Building the groundtruth and detection files"""

def getImageDetections(imagePath, weights, nms_threshold, confidenceParam, coefficient, pyramid_filter, filter_results):
    """
    Runs the detections and returns all detection into a single structure.

    Parameters
    ----------
    imagePath : str
        Path to the specific image.
    weights : str
        path to the weights.
    nms_threshold : float
        non-maximum supression threshold.
    confidenceParam : float
        confidence score for the detections (everything above this threshold is considered a valid detection).
    coefficient : int
        coefficient of the current efficientdet model (from d1 to d7).

    Returns
    -------
    detectionsList : List
        return a list with all predicted bounding-boxes.

    """
    compound_coef = coefficient
    force_input_size = None  # set None to use default size
    img_path  = imagePath

    threshold = confidenceParam
    iou_threshold = nms_threshold

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    obj_list = ['class_name']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                # replace this part with your project's anchor config
                                ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                                scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load(rootDir + 'logs/' + weights))
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

        if filter_results:
            out, pyramid_count, num_bboxes = postprocess_pyramid_filter(x,
                                                                    anchors, 
                                                                    regression, 
                                                                    classification,
                                                                    regressBoxes, 
                                                                    clipBoxes,
                                                                    threshold, 
                                                                    iou_threshold, 
                                                                    model.pyramid_limits,
                                                                    pyramid_filter)
        else:
            out, pyramid_count, num_bboxes = postprocess(x,
                                                        anchors, 
                                                        regression, 
                                                        classification,
                                                        regressBoxes, 
                                                        clipBoxes,
                                                        threshold, 
                                                        iou_threshold, 
                                                        model.pyramid_limits)

    out = invert_affine(framed_metas, out)
     
    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        detectionsList = []
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            detectionsList.append((float(out[i]['scores'][j]), x1, y1, x2, y2))
        return detectionsList, pyramid_count, num_bboxes



def generateFiles(confidenceParam, pyramid_filter, filter_results):
    """
    This method saves in two files predictions and detections for Padilla's code'
    
    Parameters
    ----------
    confidenceParam : float
        confidence threshold for the detections I want to compare against the ground truth.

    Returns
    -------
    qtyGroundTruthDets : int
        number of detections.
    qtyPredictedDets : int
        number of predictions.

    """
    
    #read config file to access path to ground truth
    params = yaml.safe_load(open(f'{rootDir}projects/{project}'+'.yml'))
    evaluationfolder = params['test_set']
    
    #Load ground truth
    dirGroundTruthPath = dirForGroundTruthAndDetections + '/' + project + '/' + str(confidenceParam) + '/groundtruths/'
    dirDetectionsPath = dirForGroundTruthAndDetections + '/' + project + '/' + str(confidenceParam) + '/detections/'
    with open(rootDir + 'datasets/' + project + '/annotations/instances_' + str(evaluationfolder) + '.json') as f:
        data = json.load(f)

    annotationsDataframe = pd.DataFrame.from_records(data['annotations'])
    imagesDataframe = pd.DataFrame.from_records(data['images'])

    qtyGroundTruthDets = 0
    qtyPredictedDets = 0
    pyramid_acc = torch.zeros(5, dtype=torch.int32)
    real_number_bboxes = 0
    for index,image in imagesDataframe.iterrows():
        imageName = image['file_name'][0:len(image['file_name'])-4]

        fileGroundTruth = open(dirGroundTruthPath + imageName + ".txt", "w")
        fileDetections = open(dirDetectionsPath + imageName + ".txt", "w")

        imageId = image['id']
        for ind,row in annotationsDataframe.loc[annotationsDataframe['image_id'] == imageId].iterrows():
            left = int(row['bbox'][0]) 
            top = int(row['bbox'][1])
            w = int(row['bbox'][2])
            h = int(row['bbox'][3])
            right = left + w
            bottom = top + h
            #<class_name> <left> <top> <right> <bottom>

            fileGroundTruth.write(project + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + "\n")
            qtyGroundTruthDets = qtyGroundTruthDets + 1
    
        fileGroundTruth.close()

        ##### Let's create the detections files to evaluate them with the Object Detection Metrics
        defaultDatasetPath = rootDir + 'datasets/' + project + '/' + evaluationfolder + '/'
        image_detections, pyramid_current, num_bboxes = getImageDetections(defaultDatasetPath + image['file_name'],
                                                                            wieghtsFileName, 
                                                                            nms_threshold, 
                                                                            confidenceParam, 
                                                                            coefficient,
                                                                            pyramid_filter,
                                                                            filter_results)
        #accumulate torch pyramid
        pyramid_acc += pyramid_current
        real_number_bboxes += num_bboxes
        if(image_detections != None):
            for confidenceScore, xd1, yd1, xd2, yd2 in image_detections:
                fileDetections.write(project + " " + str(confidenceScore) + " " + 
                                     str(xd1) + " " + str(yd1) + " " + str(xd2) + " " + str(yd2) + "\n")
                qtyPredictedDets = qtyPredictedDets + 1
        fileDetections.close()
    
    
    #returns number of detections and predictions
    print(qtyGroundTruthDets)
    print(qtyPredictedDets)
    print("------")
    return (qtyGroundTruthDets, qtyPredictedDets, pyramid_acc, real_number_bboxes)




"""## Methods to run the evaluation over the model"""

def generateResults(conf_t):
    """
    Calls Padilla's code to calculate the metrics

    Returns
    -------
    None.

    """
    print(dirForGroundTruthAndDetections + '/' + project + '/' + str(conf_t) + '/results') 
    os.system('python ' + rootDir + 'Object-Detection-Metrics/pascalvoc.py -gt ' + 
              dirForGroundTruthAndDetections + '/' + project + '/' + str(conf_t) + '/groundtruths/ -det ' + 
              dirForGroundTruthAndDetections + '/' + project + '/' + str(conf_t) + '/detections/ -t ' + 
              str(nms_threshold) + ' -gtformat xyrb -detformat xyrb -sp ' + 
              dirForGroundTruthAndDetections + '/' + project + '/' + str(conf_t) + '/results/' + ' -np')


def collectRsultsFromFiles(conf_t): 
    """
    From the results of Padilla's, get just one value from

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    precision : TYPE
        DESCRIPTION.
    recall : TYPE
        DESCRIPTION.
    f1Score : TYPE
        DESCRIPTION.

    """
    
    #read the values from the resulting run of Padilla's code
    with open(dirForGroundTruthAndDetections + '/' + project + '/' + str(conf_t) + '/results/results.txt',"r") as file1:
        FileasList = file1.readlines()

    #get what we need
    precision = FileasList[8].split()[len(FileasList[8].split()) - 1]
    recall = FileasList[9].split()[len(FileasList[9].split()) - 1]
    
    if(len(precision) > 0 and precision[1:(len(precision) - 2)] != ''):
        precision = float(precision[1:(len(precision) - 2)])
    else:
        precision = 0.0
        
    if(len(recall) > 0 and recall[1:(len(recall) - 2)] != ''):
        recall = float(recall[1:(len(recall) - 2)])
    else:
        recall = 0.0
        
    if(precision > 0.0 or recall > 0.0):
        f1Score = 2 * ((precision * recall)/(precision + recall))
    else:
        f1Score = 0.0
  
    return (FileasList[6].split()[1], FileasList[7].split()[1], precision, recall, f1Score)


#LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(cpu_list):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])


if __name__ == '__main__':
    #throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39])
    
    #create the folder to store the results
    if not os.path.exists(dirForGroundTruthAndDetections):
        os.makedirs(dirForGroundTruthAndDetections)
    if not os.path.exists(dirForGroundTruthAndDetections + '/' + project):
        os.makedirs(dirForGroundTruthAndDetections + '/' + project)
    
    if args.single_test == True:
        if project_weights != None:
            pyramid_filter = get_pyramid_mask(f'datasets/{project_weights}/train', rootDir + 'logs/' + wieghtsFileName)
        else:
            pyramid_filter = None
        
        #os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold))
        #os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/results')
        #os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/groundtruths')
        #os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/detections')
        if not os.path.exists(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold)):
            os.makedirs(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold))
            
        if not os.path.exists(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/results'):
            os.makedirs(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/results')
        if not os.path.exists(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/groundtruths'):
            os.makedirs(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/groundtruths')
        if not os.path.exists(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/detections'):
            os.makedirs(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/detections')

        if project_weights != None:
            groundTruth, detected, pyramid, filtered_num = generateFiles(args.confidence_threshold, pyramid_filter, True)
        else:
            groundTruth, detected, pyramid, filtered_num = generateFiles(args.confidence_threshold, None, False)
        generateResults(args.confidence_threshold)

        dataset, ap, precision, recall, f1Score = collectRsultsFromFiles(args.confidence_threshold)
        print('##############################################')
        print('Result over the dataset: '+ dataset)
        print('Quantity of ground truth boundig boxes: '+str(groundTruth))
        print('Quantity of predicted boundig boxes: '+str(detected))
        print('The average precision is (AP): '+ap)
        print('Precision: '+str(precision))
        print('Recall: '+str(recall))
        print('F1 Score: '+str(f1Score))
        print()
        print("----------------------")
        print("Pyramid for the test_set:")
        print(pyramid)
        #print("----------------------")
        #print("Pyramid filter:")
        #print(pyramid_filter)
        print("----------------------")
        print("Filtered num:")
        print(filtered_num)
        #prcurveImg = cv2.imread(dirForGroundTruthAndDetections+'/'+project+'/results/'+project+'.png')
        #plt.imshow(prcurveImg)
        #plt.show()
        shutil.rmtree(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/groundtruths')
        shutil.rmtree(dirForGroundTruthAndDetections + '/' + project + '/' + str(args.confidence_threshold) + '/detections')
    else:
        confidence_threshold = 0.2
        hop = 0.05
        
        if not os.path.exists(f'{dirForGroundTruthAndDetections}/{project}/{project}_results_d{coefficient}.csv'):
            with open(f'{dirForGroundTruthAndDetections}/{project}/{project}_results_d{coefficient}.csv', "w") as myfile:
                my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
                my_writer.writerow(["num_detections", "nms_threshold", 
                                    "confidence_threshold", "average_precision", 
                                    "precision", "recall", "f1_score"])

        while confidence_threshold < 1:
            #os.system('rm -r ' + dirForGroundTruthAndDetections + '/' + project + '/results')
            os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + 
                      str(confidence_threshold))
            os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + 
                      str(confidence_threshold) + '/results')
            os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + 
                      str(confidence_threshold) + '/groundtruths')
            os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project + '/' + 
                      str(confidence_threshold) + '/detections')

            
            groundTruth, detected = generateFiles(confidence_threshold)
            generateResults(confidence_threshold)
            dataset,ap,precision,recall,f1Score = collectRsultsFromFiles(confidence_threshold)
      
            ##Aqui hay que guar los valores en el CSV
            with open(f'{dirForGroundTruthAndDetections}/{project}/{project}_results_d{coefficient}.csv', "a") as myfile:
                my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
                my_writer.writerow([detected, nms_threshold, confidence_threshold,ap, precision,recall,f1Score])

            confidence_threshold = round(confidence_threshold + hop, 2)#'''
            
            #'''