# adapted from https://github.cojson_ix/efficientdet/blob/master/train.py
import argparse
import datetime
import os
import traceback
import psutil
import cv2
from shutil import copyfile

import numpy as np
import torch
import yaml
import json
from PIL import Image
import copy

from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string, preprocess_all, invert_affine, postprocess_original
from efficientdet.utils import BBoxTransform, ClipBoxes
#from utils.utils import preprocess, preprocess_all, invert_affine, postprocess_original
from utils.semi_utils import get_image_json, insert_bbox
from custom_coco_eval import run_metrics



class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)



class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False, version=1):
        super().__init__()
        self.criterion = FocalLoss(version)
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, img_names, unlabeled_names, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations, img_names, unlabeled_names, 
                                                imgs=imgs, obj_list=obj_list) 
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations, img_names, unlabeled_names)
        return cls_loss, reg_loss



def save_checkpoint(model, name, path_to_save_weights):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(path_to_save_weights, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(path_to_save_weights, name))




def main_train(opt, iteration, version):

    # Load general parameter from the model
    #----------------------------------------------------
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    path_to_save_weights = opt.saved_path + f'/{params.project_name}_{iteration}/'
    path_tensorboard = opt.log_path + f'/{params.project_name}_{iteration}/tensorboard/'
    os.makedirs(path_tensorboard, exist_ok=True)
    os.makedirs(path_to_save_weights, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}


    # Load dataset
    #----------------------------------------------------
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1356]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                               set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]),
                               iteration=iteration
                            )
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), 
                          set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)


    # Instance EfficientDet
    #----------------------------------------------------
    model = EfficientDetBackbone(num_classes = len(params.obj_list), 
                                 compound_coef = opt.compound_coef,
                                 ratios = eval(params.anchors_ratios), 
                                 scales = eval(params.anchors_scales))


    # Load last weights from COCO
    #----------------------------------------------------
    if opt.load_weights is not None:
        weights_path = opt.load_weights

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        #print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        #Random initialization
        #print('[Info] initializing weights...')
        #init_weights(model)
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        print("error 1")
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        return None


    # Freeze backbone if train head_only
    #----------------------------------------------------
    '''
    if opt.head_only:
        print("THIS SHOULD BE SET TO FALSE SINCE THIS IS SEMISUPERVISED")
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')
    '''

    # Get image names
    #----------------------------------------------------
    unlabeled_images_set = params.train_set_unlabeled 
    root_dir_images_unlabeled = f'datasets/{params.project_name}/{unlabeled_images_set}'
    unlabeled_names = os.listdir(root_dir_images_unlabeled)


    # Set a batch normalization option when using multiple GPU's
    #----------------------------------------------------
    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False


    # Include on top of the model the loss function - Focal loss
    # This way to reduce the memory usage on gpu0 and speedup
    #----------------------------------------------------
    model = ModelWithLoss(model, debug=opt.debug, version=version)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)


    # Set the optimizer and a scheduler to tune it
    #----------------------------------------------------
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


    # Basic variables to train
    #----------------------------------------------------
    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = 0
    step = max(0, last_step)
    num_iter_per_epoch = len(training_generator)
    writer = SummaryWriter(path_tensorboard + f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')


    # Enable the model in learning model
    #----------------------------------------------------
    model.train()


    # Loop over the epochs
    #----------------------------------------------------
    try:
        for epoch in range(opt.num_epochs):

            # In case the weights are loaded from a previous training
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue
            

            # Perform a batch training
            #------------------------
            #------------------------
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):

                # In case the weights are loaded from a previous training
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue

                try:
                    # Load data
                    #----------
                    imgs = data['img']
                    annot = data['annot']
                    img_names = data['img_names']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()
                    #----------


                    # Calculate the loss
                    #----------
                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, img_names, unlabeled_names, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue
                    #----------


                    # Calculate the gradients
                    #----------
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    #----------


                    # Update weights
                    #----------
                    optimizer.step()
                    #----------


                    # Register log and print in console progress
                    #----------
                    epoch_loss.append(float(loss))
                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1
                    if step % opt.save_interval == 0 and step > 0:
                        #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_trained_weights_semi.pth', path_to_save_weights)
                        #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}_semi.pth', path_to_save_weights)
                        print('checkpoint...')

                        #with open(os.path.join(path_to_save_weights, f"best_epoch-d{opt.compound_coef}_semi.txt"), "a") as my_file: 
                        #    my_file.write(f"Checkpoint-Epoch:{epoch} / Step: {step} / Loss: {best_loss}\n") 
                    #----------

                except Exception as e:
                    #Error performing one batch training
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            
            # Tuneup the optimizer
            # -> ReduceLROnPlateau using the optimizer
            scheduler.step(np.mean(epoch_loss))
            #------------------------
            #------------------------



            #Perform a validation calculation
            #------------------------
            #------------------------
            if epoch % opt.val_interval == 0:
                
                # Disable learning mode of the model
                #----------
                model.eval()


                # Loop over the validation set
                #----------
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    
                    # Double check to disable learning mode
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']
                        img_names = data['img_names']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, img_names, unlabeled_names, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())
                #----------


                # Get loss
                #----------
                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss
                #----------


                # Log the results
                #----------
                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                

                # Save checkpoint only if there is an improvement
                #----------
                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_trained_weights_semi.pth', path_to_save_weights)
                    with open(os.path.join(path_to_save_weights, f"best_epoch-d{opt.compound_coef}_semi.txt"), "a") as my_file: 
                        my_file.write(f"Epoch:{epoch} / Step: {step} / Loss: {best_loss}\n") 
                #----------


                # Enable the model again to learn
                #----------
                model.train()
                #----------


                # Early stopping
                #----------
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
                #----------
            #------------------------
            #------------------------


    except KeyboardInterrupt:
        # in case of exception, save current model
        #save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}_semi.pth')
        writer.close()
    #----------------------------------------------------
    writer.close()


    # Disable model to learn and return it
    #model.eval()
    #return model'''





def generate_pseudolabels(opt, iteration, label_confidence):
    #control print
    print('1 - setting up parameters.')

    #important parameters 
    params = Params(f'projects/{opt.project}.yml')

    project_name = params.project_name
    compound_coef = opt.compound_coef
    force_input_size = None  # set None to use default size
    threshold = 0.4
    iou_threshold = 0.4
    #obj_list = ['apple']
    obj_list = params.obj_list

    ratios_ = eval(params.anchors_ratios)#[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    scales_ = eval(params.anchors_scales)#[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    #weights_ = f'logs/apple_semi_annotated_{str(iteration)}/efficientdet-d4_trained_weights_semi.pth'
    #weights_ = f'logs/apple_semi_annotated/efficientdet-d4_trained_weights.pth'
    weights_ = f'logs/{opt.project}_{str(iteration)}/efficientdet-d{opt.compound_coef}_trained_weights_semi.pth'
    unlabeled_images_set = params.train_set_unlabeled #'valid'
    labeled_images_set = params.train_set #'train'
    
    #Get images
    root_dir_images_unlabeled = f'datasets/{project_name}/{unlabeled_images_set}'
    original_names_unlabeled = os.listdir(root_dir_images_unlabeled)
    img_path_unlabeled = [root_dir_images_unlabeled + '/' + i for i in original_names_unlabeled]

    destination_dir_images_annotated = f'datasets/{project_name}/{labeled_images_set}_{str(iteration)}_results/'
    if not os.path.exists(destination_dir_images_annotated):
        os.mkdir(destination_dir_images_annotated)

    #create a new folder with the labeled images
    destination_labeled = f'datasets/{project_name}/{labeled_images_set}_{str(iteration)}/'
    if not os.path.exists(destination_labeled):
        os.mkdir(destination_labeled)

    #copy all images from train/labeled samples, into the iteration folder
    train_labeled_images = f'datasets/{project_name}/' + labeled_images_set
    for i in os.listdir(train_labeled_images):
        copyfile(train_labeled_images + '/' + i, destination_labeled + '/' + i)

    
    #Get original JSON
    root_dir_orig_json = f'datasets/{project_name}/annotations/instances_{labeled_images_set}.json'
    new_json_path = f'datasets/{project_name}/annotations/instances_{labeled_images_set}_{str(iteration)}.json'
    
    
    #Model variables
    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    
    #control print
    print('2 - Get predictions.')

    #GET predictions -> prepare the model
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess_all(img_path_unlabeled, max_size=input_size)
    
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
        for i in range(int(x.shape[0])+1):
            try:
                _, r, c, a = model(x[i:i+1])
            
                if i == 0:
                    regression = r
                    classification = c
                    anchors = a
                else:
                    regression = torch.cat([regression, r], dim=0)
                    classification = torch.cat([classification, c], dim=0)
                    anchors = torch.cat([anchors, a], dim=0)
                #print(i)
            except RuntimeError as e:
                print("Error using this image")
    
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
        new_json = copy.deepcopy(orig_json)
        update_json_file = False

        #control print
        print('3 - Keep prediction only above a threshold.')

        #Iter over ALL unlabeled images
        for i in range(len(ori_imgs)):
            #do nothing if there are no rois
            if len(out[i]['rois']) == 0:
                continue
            
            #Get a temporal JSON with the current image already inserted
            temporal_json = copy.deepcopy(new_json)
            temporal_json, image_dict = get_image_json(json_=temporal_json, img_name=original_names_unlabeled[i])
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
                new_json = copy.deepcopy(temporal_json)
            
                #save image with all detected bboxes
                image_ = Image.fromarray(np.uint8(ori_imgs[i]), 'RGB')
                image_.save(destination_dir_images_annotated + original_names_unlabeled[i])

                #copy image to labeled images
                current_img = cv2.imread(root_dir_images_unlabeled + '/' + original_names_unlabeled[i])
                cv2.imwrite(destination_labeled + original_names_unlabeled[i], current_img) 
            #break

    #control print
    print('4 - Save new JSON.')
    #save new json file
    if update_json_file:
        with open(new_json_path, 'w') as json_file:
            json.dump(new_json, json_file)
    else:
        with open(new_json_path, 'w') as json_file:
            json.dump(orig_json, json_file)


    print('5 - Done.')#'''


def measure_performance(opt, iteration, nms_threshold, confidence_threshold):
    #------------------------------------------------------------------------------------------------------------------------------    
    project_name = opt.project
    weights_path = weights_ = f'logs/{opt.project}_{str(iteration)}/efficientdet-d{opt.compound_coef}_trained_weights_semi.pth'
    compound_coef = opt.compound_coef
    use_cuda = False#True if torch.cuda.is_available() else False
    use_float16 = False
    override_prev_results = True
    max_detections = 100000
    #------------------------------------------------------------------------------------------------------------------------------

    run_metrics(compound_coef, nms_threshold, 
                use_cuda, use_float16, 
                override_prev_results, project_name, 
                weights_path, confidence_threshold, 
                max_detections, iteration)


def get_args():
    parser = argparse.ArgumentParser('EfficientDet Pytorch')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


#LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(cpu_list):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])



if __name__ == '__main__':
    throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39]) 
    
    opt = get_args()
    nms_threshold = 0.5
    confidence_threshold = 0.4
    iterations = 5
    version = 1
    
    
    for i in list(range(iterations)):
        iteration = i + 1
        main_train(opt, iteration, version)
        generate_pseudolabels(opt, iteration, 0.95)
        measure_performance(opt, iteration, nms_threshold, confidence_threshold)#'''
        #break
    #generate_pseudolabels(opt, 1, 0.95)
