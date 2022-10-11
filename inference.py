import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Path from dir of model and images
img_p = 'colondataset/valid'
model_p = 'models/colon_resnet18.pth'

#   Loading model
model = torch.load(model_p,map_location=device)
model = model.eval()

#   Class labels for prediction
class_names = ['anus', 'blank', 'cecum']

#   Preprocessing transformations
preprocess=transforms.Compose([
        transforms.Resize(size=512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

#   Loading image

#   Inference
imgs = os.listdir(img_p)
for i in imgs:
    if i == '.DS_Store': continue
    i_path = os.path.join(img_p, i)
    img = Image.open(i_path).convert('RGB')
    inputs = preprocess(img).unsqueeze(0).to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)    
    label = class_names[preds]

    #   print output
    print('{} is {}'.format(i_path, label))


