# Yolov3-facial-landmark-detection

This repository contains files for training, testing and inference Yolov3 for multi-task face detection and facial landmarks extraction.

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

1. In order to evaluate the models, first download Widerface Validation dataset from [Widerface website](http://shuoyang1213.me/WIDERFACE/) and WFLW dataset from [website](https://wywu.github.io/projects/LAB/WFLW.html) or [Google drive](https://wywu.github.io/projects/LAB/WFLW.html). 

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


### Facial Landmarks Extraction



## References
- [yolo-face-with-landmark](https://github.com/ouyanghuiyu/yolo-face-with-landmark)
- [yolov3](https://github.com/ultralytics/yolov3)
- [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
