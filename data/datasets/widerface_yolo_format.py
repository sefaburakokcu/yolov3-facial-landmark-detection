'''The code is retrieved from
https://github.com/ouyanghuiyu/yolo-face-with-landmark/blob/master/src/retinaface2yololandmark.py
and refactored for better usage and visualization.

'''

import os
import cv2
from tqdm import tqdm

def get_widerface_data(input_labels_path):
    with open(input_labels_path, "r") as f:
        lines = f.readlines()
        isFirst = True
        labels = []
        words = []
        imgs_path = []
        for line in tqdm(lines, desc="labels are loading"):
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[1:].strip()
                path = os.path.join(input_labels_path.replace(input_labels_path.split("/")[-1], "images"), path)
        
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        
        words.append(labels)
        return words, imgs_path
    

def save_widerface_yolo(words, imgs_path, save_labels_folder):
    image_count = 0
    for path,word in tqdm(zip(imgs_path,words), total=len(imgs_path), desc="labels are converting to yolo format"):
        img = cv2.imread(path)
        img_height = img.shape[0]
        img_width = img.shape[1]
        rel_bbox_list = []
    
        for anno in word:
    
            landmark =  []
            for zu in [[4,5],[7,8],[10,11],[13,14],[16,17]]:
                if anno[zu[0]] == -1:
                    landmark.append("-1")
                    landmark.append("-1")
                else:
                    landmark.append(str(float(anno[zu[0]] * 1.0  / img_width)))
                    landmark.append(str(float(anno[zu[1]] * 1.0  / img_height)))
    
    
            x1, y1, w, h = anno[:4]
            if w < 10 or h < 10: # Face size check (discard if smaller than 100 pixels)
#                img[int(y1):int(y1+h),int(x1):int(x1+w),:] = 127
                continue
            
            rel_cx = str(float((x1 + int(w/2)) / img_width))
            rel_cy = str(float((y1 + int(h/2)) / img_height))
            rel_w = str(float(w / img_width))
            rel_h = str(float(h / img_height))
    
            string_bbox = "0 " + rel_cx + " " + rel_cy + " " + rel_w + " " + rel_h + " " + " ".join(landmark)
            rel_bbox_list.append(string_bbox)
        image_count += 1
    
        save_image_name = "wider_" + str(image_count)
    
        cv2.imwrite(save_labels_folder + save_image_name + ".jpg", img)
        with open(save_labels_folder + save_image_name + ".txt", "w") as f:
            for i in rel_bbox_list:
                f.write(i + "\n")
                
                
if __name__ == '__main__':
    input_labels_path = "/home/sefa/data/widerface/train/label.txt"
    save_labels_folder = "/home/sefa/data/widerface/train/yololandmark_wider_train/"
    
    if not os.path.exists(save_labels_folder):
        os.makedirs(save_labels_folder)
        
    words, imgs_path = get_widerface_data(input_labels_path)
    save_widerface_yolo(words, imgs_path, save_labels_folder)
    
    
    