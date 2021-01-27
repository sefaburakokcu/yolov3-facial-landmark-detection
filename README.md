# Yolov3-facial-landmark-detection

This repository contains files for training and testing Yolov3 for multi-task face detection and facial landmarks extraction.

![Example Output](https://github.com/sefaburakokcu/yolov3-facial-landmark-detection/blob/main/outputs/extra/2_Demonstration_Demonstration_Or_Protest_2_58.jpg)

**P.S.** A jupyter-notebbok for all parts can be found [here](https://github.com/sefaburakokcu/yolov3-facial-landmark-detection/blob/main/src/yolov3_main.ipynb)

## Installation

* First, clone the repository.

```bash
git clone https://github.com/sefaburakokcu/yolov3-facial-landmark-detection.git
```

* Then, install prerequisites.

```bash
pip install -r requirements.txt
```


## Training

1. For training the models with Widerface dataset, first download dataset from [Widerface website](http://shuoyang1213.me/WIDERFACE/). Then, under *data/datasets/* run,

```bash
python widerface_yolo_format.py
```

Or downlaod Widerface tarining dataset in YOLO format directly from [Google Drive](https://drive.google.com/file/d/1VYxoZetzbvLysGbUYbAMTF5FepXocjDj/view?usp=sharing).

2. Under *src* folder, run

```bash
python train.py
```

## Inference

For inference, pretrained weights can be used. Pretrained weights can be download from [Google Drive](https://drive.google.com/file/d/1_gVszd6i7LtiaTTiOj_zef91Qz-ehGDE/view?usp=sharing).

Under *src* folder, run

```bash
python inference.py
```

## Tests

1. In order to evaluate the models, first download Widerface Validation dataset from [Widerface Website](http://shuoyang1213.me/WIDERFACE/) and WFLW dataset from [WFLW Website](https://wywu.github.io/projects/LAB/WFLW.html) or [Google Drive](https://drive.google.com/file/d/1dtFIHkMc9H-9NjbRvqSsbc0bzDFlkdia/view?usp=sharing). 

2. Then, under *src* run,

```bash
python test.py
```

in order to save face detection and facial landmarks predictions. 

3. Finally, under *src/evaluations/widerface/*, run

```bash
python evaluate_widerface.py
```
for face detection performance and under *src/evaluations/wiflw/*, run

```bash
python evaluate_wflw.py
```
for facial landmarks extraction performance.

### Face Detection

Evaluation of models on Widerface Validation dataset for face detection is indicated below.
Average Precision is used as a performance metric.

| Models  | Easy | Medium | Hard |
| ------------- | ------------- | ------------- | ------------- |
| Mobilenetv2(0.75)  | 0.85 | 0.83  | 0.63 |
| Mobilenetv2(1.0)  | 0.87  | 0.86  | 0.69  |
| [Retinaface(Mobilenetv2(0.25))](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)  | 0.90  | 0.87  | 0.67 |
| [Retinaface(Resnet50)](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)  | 0.93  | 0.91  | 0.69 |
| MTCNN  | 0.79  | 0.76  | 0.50  |


### Facial Landmarks Extraction

Evaluation of models on WFLW dataset for facial landmarks extraction is shown below.
Average Root Mean Square Error(RMSE) is chosen as a performance metric.

| Models  | RMSE |
| ------------- | ------------- |
| Mobilenetv2(0.75)  | 6.53 |
| Mobilenetv2(1.0)  | 4.36  | 
| [Retinaface(Mobilenetv2(0.25))](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)  | 4.03  | 
| [Retinaface(Resnet50)](https://github.com/deepinsight/insightface/tree/master/detection/RetinaFace)  | 0.93  | 3.22  |
| MTCNN  | 4.5  |


## References
- [yolo-face-with-landmark](https://github.com/ouyanghuiyu/yolo-face-with-landmark)
- [yolov3](https://github.com/ultralytics/yolov3)
- [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
