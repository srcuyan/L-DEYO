## Introduction

L-DEYO is an optimized and lightweight object detection model designed specifically for the challenging task of intelligent coal gangue recognition. Coal gangue is a byproduct of the coal mining process and requires efficient identification and separation to reduce environmental impact and improve resource recovery. Traditional methods often struggle with accuracy and real-time performance in industrial settings, particularly when dealing with complex textures and small, overlapping objects.

L-DEYO addresses these challenges by integrating key improvements in model architecture, including:

- **Ghost Module** in the backbone, which reduces computational complexity while maintaining feature extraction performance.
- **BiFPN (Bidirectional Feature Pyramid Network)** for multi-scale feature extraction, enhancing detection across different object sizes.
- **Adaptive Focus Attention Decoder**, optimizing the detection of small and densely packed objects.
- **Custom Weighted Loss Function**, tackling class imbalance and improving classification accuracy in complex environments.

Through extensive experiments, L-DEYO demonstrates superior accuracy and real-time performance compared to existing methods, making it highly suitable for deployment in industrial applications where resource efficiency and precision are critical.

This repository contains the full implementation of L-DEYO, including pre-trained models, training scripts, and detailed documentation to help users reproduce and extend the results.

## Key Features

- Lightweight architecture optimized for real-time industrial applications.
- Improved handling of complex textures and overlapping objects.
- Multi-scale feature extraction with BiFPN.
- Custom loss function for addressing class imbalance.

Feel free to explore the code and contribute to the project!

## Install
```bash
pip install ultralytics
```

## Step-by-step Training

#### Frist Training Stage
Replace `ultralytics/engine/trainer.py` with `trainer.py`

```python
from ultralytics import YOLO

# Train from Scratch
model = YOLO("cfg/models/v10/yolov10n.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.5, mixup = 0, copy_paste = 0)

# Train from Scratch
model = YOLO("cfg/models/v9/yolov9c.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.9, mixup = 0.15, copy_paste = 0.3)

# Train from Scratch
model = YOLO("cfg/models/v9/yolov9e.yaml")

# Use the model
model.train(data = "coco.yaml", epochs = 500, scale = 0.9, mixup = 0.15, copy_paste = 0.3)
```

#### Second Training Stage

Please note that if you directly adopt a model pre-trained on the COCO dataset, you may not achieve the best results, as we have set the one-to-many branch to a frozen state, which means the one-to-many branch will not undergo further fine-tuning optimization based on your dataset. You will need to fine-tune the model pre-trained on the COCO dataset during the First Training Stage.

If you have made changes to the `ultralytics/engine/trainer.py` during the frist training stage, please revert it.

```python
from ultralytics import DEYO

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5n.yaml")
model.load("best-n.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 144, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.5, mixup = 0, copy_paste = 0, freeze = 23)

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5c.yaml")
model.load("best-c.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 22)

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5e.yaml")
model.load("best-e.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 42)
```

## Multi GPUs
The multi-GPU setup for the first stage of training is consistent with YOLOv8. For the second stage of training, you need to follow the following steps:

Replace `ultralytics/engine/trainer.py` with our modified `ddp/trainer.py`
```bash
rm -rf Path/ultralytics
cp -r ultralytics Path/  # Path：The location of the ultralytics package
```

```python
import torch
from ultralytics import DEYO

# Load a model
model = DEYO("cfg/models/deyo/deyov1.5e.yaml")
model.load("best-e.pt")
torch.save({"epoch":-1, "model": model.model.half(), "optimizer":None}, "init.pt")
model = DEYO("init.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 72, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 0, scale = 0.9, mixup = 0.15, copy_paste = 0.3, freeze = 42, device = '0, 1, 2, 3, 4, 5, 6, 7')
```

