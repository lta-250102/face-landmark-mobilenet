from ultralytics import YOLO
from ultralytics.engine.results import Results
from PIL import Image
import requests

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')[0]  # predict on an image

import numpy as np
print('---------------------------------------------')
print(results.boxes)