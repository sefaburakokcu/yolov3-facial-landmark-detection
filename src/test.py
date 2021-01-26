import math
import os
import sys
import numpy as np
import cv2
import torch
from models import  *
from utils.torch_utils import select_device
from utils.utils import non_max_suppression
import cv2
import numpy as  np
import glob
from hyp import  hyp


def get_model(net_type, weights, device=None):
    assert net_type in ["mbv2_1", "mbv2_75", "darknet53"]
    if device is None:
    	device = select_device('cpu')

    if net_type.startswith("mbv2_1"):
        backbone = mobilenet_v2(pretrained=False, width_mult = 1.0)
    elif net_type.startswith("mbv2_75"):
        backbone = mobilenet_v2(pretrained=False, width_mult = 0.75)        
    elif net_type.startswith("darknet53"):
        backbone = DarkNet53()
        
    if 'light' in net_type:
        net = DarknetWithShh(backbone, hyp, light_head=True).to(device)
    else:
        net = DarknetWithShh(backbone, hyp).to(device)
        
    net.load_state_dict(torch.load(weights, map_location=device)['model'])
    net.eval()
    return net


def save_widerface_predictions(net, val_image_root, val_results_root="../outputs/widerface_results/", long_side=640):
    counter = 0
    device = next(net.parameters()).device
    for parent, dir_names, file_names in os.walk(val_image_root):
        for file_name in file_names:
            if not file_name.lower().endswith('jpg'):
                continue
            orig_image = cv2.imread(os.path.join(parent, file_name))
            ori_h, ori_w, _ = orig_image.shape
            LONG_SIDE = long_side
            if long_side == -1:
                max_size = max(ori_w, ori_h)
                LONG_SIDE = max(32, max_size - max_size % 32)
    
            if ori_h > ori_w:
                scale_h = LONG_SIDE / ori_h
                tar_w = int(ori_w * scale_h)
                tar_w = tar_w - tar_w % 32
                tar_w = max(32, tar_w)
                tar_h = LONG_SIDE
            else:
                scale_w = LONG_SIDE / ori_w
                tar_h = int(ori_h * scale_w)
                tar_h = tar_h - tar_h % 32
                tar_h = max(32, tar_h)
                tar_w = LONG_SIDE
    
            scale_w = tar_w * 1.0 / ori_w
            scale_h = tar_h * 1.0 / ori_h
    
            image = cv2.resize(orig_image, (tar_w, tar_h))
    

            image = image[..., ::-1]
            image = image.astype(np.float32)
            image = image / 255.0
            image = np.transpose(image, [2, 0, 1])
    
            image = np.expand_dims(image, axis=0)
    
            image = torch.from_numpy(image)
            image = image.to(device)
            pred = net(image)[0]
    
            pred = non_max_suppression(pred, 0.01, 0.35,
                                       multi_label=False, classes=0, agnostic=False, land=True, point_num=hyp['point_num'])
            boxes = []
            if pred[0] is not None:
                det = pred[0].cpu().detach().numpy()
                orig_image = orig_image.astype(np.uint8)
                det[:, :4] = det[:, :4] / np.array([scale_w, scale_h] * 2)
                det[:, 5:15] = det[:, 5:15] / np.array([scale_w, scale_h] * 5)
    
                for detection in det:
                    boxes.append(detection[:5].tolist())
    
            event_name = parent.split('/')[-1]
            if not os.path.exists(os.path.join(val_results_root, event_name)):
                os.makedirs(os.path.join(val_results_root, event_name))
            fout = open(os.path.join(val_results_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
    
            fout.write(file_name.split('.')[0] + '\n')
            fout.write(str(len(boxes)) + '\n')
            for i in range(len(boxes)):
                bbox = boxes[i]
    
                fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), bbox[4] if bbox[4] <= 1 else 1) + '\n')
            fout.close()
            counter += 1
            if counter % 250 == 0: 
                print('[%d] %s is processed. detect: %d' % (counter, file_name,len(boxes)))


def save_wflw_predictions(net, val_labels_path, val_image_root, val_results_root="../outputs/wflw_results/", long_side=640):
    counter = 0
    device = next(net.parameters()).device
    
    with open(val_labels_path, "r") as f:
        labels = f.read().splitlines()
        
        for label in labels:
            label = label.split(' ')
            label_name = label[-1]
            file_name = label_name.split('/')[-1]
            image_path = os.path.join(val_image_root, label_name)
        
            orig_image = cv2.imread(image_path)
            ori_h, ori_w, _ = orig_image.shape
            LONG_SIDE = long_side
            if long_side == -1:
                max_size = max(ori_w, ori_h)
                LONG_SIDE = max(32, max_size - max_size % 32)
    
            if ori_h > ori_w:
                scale_h = LONG_SIDE / ori_h
                tar_w = int(ori_w * scale_h)
                tar_w = tar_w - tar_w % 32
                tar_w = max(32, tar_w)
                tar_h = LONG_SIDE
            else:
                scale_w = LONG_SIDE / ori_w
                tar_h = int(ori_h * scale_w)
                tar_h = tar_h - tar_h % 32
                tar_h = max(32, tar_h)
                tar_w = LONG_SIDE
    
            scale_w = tar_w * 1.0 / ori_w
            scale_h = tar_h * 1.0 / ori_h
    
            image = cv2.resize(orig_image, (tar_w, tar_h))
    

            image = image[..., ::-1]
            image = image.astype(np.float32)
            image = image / 255.0
            image = np.transpose(image, [2, 0, 1])
    
            image = np.expand_dims(image, axis=0)
    
            image = torch.from_numpy(image)
            image = image.to(device)
            pred = net(image)[0]
    
            pred = non_max_suppression(pred, 0.01, 0.35,
                                       multi_label=False, classes=0, agnostic=False, land=True, point_num=hyp['point_num'])
            boxes = []
            landmarks = []
            if pred[0] is not None:
                det = pred[0].cpu().detach().numpy()
                orig_image = orig_image.astype(np.uint8)
                det[:, :4] = det[:, :4] / np.array([scale_w, scale_h] * 2)
                det[:, 5:15] = det[:, 5:15] / np.array([scale_w, scale_h] * 5)
    
                for detection in det:
                    boxes.append(detection[:5].tolist())
                    landmarks.append(detection[5:15].tolist())
                    
            event_name = label_name.split('/')[0]
            if not os.path.exists(os.path.join(val_results_root, event_name)):
                os.makedirs(os.path.join(val_results_root, event_name))
            fout = open(os.path.join(val_results_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
    
            fout.write(file_name.split('.')[0] + '\n')
            fout.write(str(len(boxes)) + '\n')
            for i in range(len(boxes)):
                bbox = boxes[i]
                landmark = landmarks[i]
                fout.write('%d %d %d %d %.03f %d %d %d %d %d %d %d %d %d %d' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), bbox[4] if bbox[4] <= 1 else 1, 
                           math.floor(landmark[0]), math.floor(landmark[1]), math.floor(landmark[2]), math.floor(landmark[3]), math.floor(landmark[4]), 
                           math.floor(landmark[5]), math.floor(landmark[6]), math.floor(landmark[7]), math.floor(landmark[8]), math.floor(landmark[9]))+ '\n')
            fout.close()
            counter += 1
            if counter % 250 == 0: 
                print('[%d] %s is processed. detect: %d' % (counter, file_name,len(boxes)))
                
if __name__ == '__main__':
    net_type = "mbv2_1"
    weights = f"../weights/{net_type}_best.pt"
    
    val_image_root = "/home/sefa/Downloads/WFLW_images/"
    val_results_root = f"../outputs/wlfw_results/{net_type}/"
    val_labels_path = "/home/sefa/Downloads/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    
    device = select_device('0')
    
    net = get_model(net_type, weights, device)
#    save_widerface_predictions(net, val_image_root, val_results_root)
    
    save_wflw_predictions(net, val_labels_path, val_image_root, val_results_root)