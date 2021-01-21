
import cv2
import glob
import random
import argparse
import torch
import numpy as  np

from pathlib import Path
from hyp import  hyp
from models import  *
from utils.torch_utils import select_device, time_synchronized
from utils.utils import non_max_suppression, xyxy2xywh
from utils.general import scale_coords, plot_one_box, \
                          check_requirements, plot_one_landmarks
from utils.inference_datasets import LoadImages, LoadWebcam



def get_model(net_type, weights, device):
    assert net_type in ['mbv3_small_1', 'mbv3_small_75', 'mbv3_large_1', 'mbv3_large_75',
                   "mbv3_large_75_light", "mbv3_large_1_light", 'mbv3_small_75_light', 'mbv3_small_1_light',
                   "mbv2_1", "mbv2_75",
                   ]
    
    if net_type.startswith("mbv3_small_1"):
        backbone = mobilenetv3_small()
    elif net_type.startswith("mbv3_small_75"):
        backbone = mobilenetv3_small( width_mult=0.75)
    elif net_type.startswith("mbv3_large_1"):
        backbone = mobilenetv3_large()
    elif net_type.startswith("mbv3_large_75"):
        backbone = mobilenetv3_large( width_mult=0.75)
    elif net_type.startswith("mbv3_large_f"):
        backbone = mobilenetv3_large_full()
    elif opt.net.startswith("mbv2_1"):
        backbone = mobilenet_v2(pretrained=False, width_mult = 1.0)
    elif opt.net.startswith("mbv2_75"):
        backbone = mobilenet_v2(pretrained=False, width_mult = 0.75)        

    if 'light' in net_type:
        net = DarknetWithShh(backbone, hyp, light_head=True).to(device)
    else:
        net = DarknetWithShh(backbone, hyp).to(device)
        
    net.load_state_dict(torch.load(weights, map_location=device)['model'])
    net.eval()
    return net


def run_inference():
    device = select_device(opt.device)
    model = get_model(opt.net, opt.weights, device=device)
    
    webcam = opt.input.isnumeric() or  opt.input.lower().startswith(('rtsp://'))

    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.input, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.input, img_size=opt.img_size)
        
    point_num = hyp['point_num']
    names = ["face", "no_face"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for path, img, im0s, vid_cap in dataset:
        img = img.astype(np.float64)
        img = torch.from_numpy(img).to(device).float()
        
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        t1 = time_synchronized()
        pred = model(img)[0]
    
        pred = non_max_suppression(pred,opt.conf_thres, 0.35,
                                   multi_label=False, classes=0, agnostic= False,land=True ,point_num=point_num)
      
        t2 = time_synchronized()
        
            # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
    
            p = Path(p)  # to Path
            save_path = opt.save_dir +"/" + p.name  # img.jpg
            
            s += '%gx%g ' % img.shape[2:]  # print string
           
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                det[:, 5:5+point_num*2] = scale_coords(img.shape[2:], det[:, 5:5+point_num*2], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string
    
                # Save and show results
                for (*xyxy, conf), (*xyxyxyxyxy,cls) in zip(reversed(det[:, :5]), reversed(det[:, 5:])):
                    if opt.save_img or opt.view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        plot_one_landmarks(xyxyxyxyxy, im0)
                        
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
    
            # Stream results
            if opt.view_img:
                cv2.imshow(str(p), im0)
    
            # Save results (image with detections)
            if opt.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
    
                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)


    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='mbv2_1', help='net')
    parser.add_argument('--weights', type=str, default='../weights/mbv2_1_last.pt', help='initial weights path')
    parser.add_argument('--input', type=str, default='../data/inputs/images/', help='a image folder or a video or webcam')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=False,  help='display results')
    parser.add_argument('--save-img', action='store_true',default=True, help='save results')
    parser.add_argument('--save-dir', type=str, default='../data/outputs/', help='save path')
    opt = parser.parse_args()
    
    print(opt)
#☻    check_requirements()
    
    with torch.no_grad():
        run_inference()