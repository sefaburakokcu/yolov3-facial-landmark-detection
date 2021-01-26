
import os
import tqdm
import pickle
import argparse
import cv2
import numpy as np
from scipy.io import loadmat
from box_overlaps  import bbox_overlaps
from IPython import embed


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    preds = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] is '':
            continue
        if len(line) != 15:
            print(img_file)
            continue
        preds.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]),
                          float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]),
                          float(line[10]), float(line[11]), float(line[12]), float(line[13]), float(line[14])])
    preds = np.array(preds)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], preds


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    landmarks = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _landmarks = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _landmarks
        landmarks[event] = current_event
    return landmarks


def get_gts(gt_file):
    with open(gt_file, 'r') as f:
        lines = f.readlines()

    preds = []
    landmarks = dict()
    pbar = tqdm.tqdm(lines)
    pbar.set_description('Reading Ground Truth Labels ')
    for line in pbar:
        line = line.rstrip('\n').split(' ')
        event_name, image_name = (line[-1]).split('/')
        
        if event_name not in landmarks.keys():
            event_dict = dict()
        else:
            event_dict = landmarks[event_name]
            
        if image_name.rstrip('.jpg') not in  event_dict.keys():
            image_labels_list= []
        else:
            image_labels_list = event_dict[image_name.rstrip('.jpg')]
        
        new_image_labels_list = [[float(line[-11]), float(line[-10]), float(line[-9])-float(line[-11]), float(line[-8])-float(line[-10]),
                                 float(line[192]), float(line[193]), float(line[194]), float(line[195]), float(line[108]),
                                 float(line[109]), float(line[176]), float(line[177]), float(line[184]), float(line[185])]]
        
        image_labels_list = image_labels_list + new_image_labels_list
        
        event_dict[image_name.rstrip('.jpg')] = image_labels_list
        landmarks[event_name] = event_dict
    
    for event_name in landmarks.keys():
        for image_name in landmarks[event_name].keys():
            landmarks[event_name][image_name] = np.array(landmarks[event_name][image_name])
        
    return landmarks


def image_eval(pred, gt, iou_thresh=0.5):
    _pred = pred.copy()
    _gt = gt.copy()
    
    pred_match_gt = np.zeros(_pred.shape[0])


    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
       
        if max_overlap >= iou_thresh:
            pred_match_gt[h] = max_idx
        else:
            pred_match_gt[h] = -1
    return pred_match_gt.astype(np.int)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mse(predictions, targets):
    return (((predictions - targets) ** 2).mean())

    
def evaluate_wflw(pred, gt):
    pred = get_preds(pred)
    gt = get_gts(gt)
    
    event_list = list(pred.keys())
    event_num = len(event_list)

    error_list = []

    pbar = tqdm.tqdm(range(event_num))
    for i in pbar:
        pbar.set_description('Processing')
        event_name = str(event_list[i])
        
        pred_list = pred[event_name]
        gt_list = gt[event_name]
        
        for j in gt_list.keys():
            pred_info = pred_list[j]
            gt_info = gt_list[j]
         
            if len(gt_info) == 0 or len(pred_info) == 0:
                continue
            
            pred_bboxes = pred_info[:, :5]
            gt_bboxes = gt_info[:, :4]
            
            pred_match_gt = image_eval(pred_bboxes, gt_bboxes)
            
            for preds, idx in zip(pred_info, pred_match_gt):
                if idx == -1:
                    continue
                
                pred_landmarks = preds[5:]
                gt_landmarks = gt_info[idx, 4:]
                
                
                error = rmse(pred_landmarks, gt_landmarks)
                error_list.append(error)
                


    print("\n==================== Results ====================")
    print(f"Average RMSE: {np.array(error_list).mean()}")
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="/home/sefa/workspace/projects/face_projects/face_detection/after_07092020/yolov3-facial-landmark-detection-main/outputs/wlfw_results/mbv2_75/")
    parser.add_argument('-g', '--gt', default='/home/sefa/Downloads/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt')

    args = parser.parse_args()
    evaluate_wflw(args.pred, args.gt)












