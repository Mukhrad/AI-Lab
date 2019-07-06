import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable

import json
import argparse

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Load a model from a checkpoint and make a prediction on an image")
    parser.add_argument('--img_path', type=str, help='set the image path')
    parser.add_argument('--topk', default=5, type=int, help='set the num of topk')
    parser.add_argument('--gpu', default=False, type=bool, help='set the gpu mode')
    parser.add_argument('--checkpoint', type=str, help='set the checkpoint path')
    args = parser.parse_args()
    return args


def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    saved_model = checkpoint['model']
    saved_optimizer = checkpoint['optimizer']
    saved_epoch = checkpoint['epoch']
    
    return saved_model, saved_optimizer, saved_epoch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    # Resize to 256
    image = image.resize((256, 256))
    # Center crop to 224
    image = image.crop((16, 16, 240, 240))
    
    # 0-255 to 0-1
    image = np.array(image)
    image = image/255.
    
    # Nomalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Transpose
    image = np.transpose(image, (2, 0, 1))
    
    return image.astype(np.float32)


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img = Image.open(image_path)
    img = process_image(img)
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    
    
    img = torch.from_numpy(img)
    
    model.eval()
    inputs = img.to(device)
    logits = model.forward(inputs)
    
    ps = F.softmax(logits,dim=1)
    topk = ps.cpu().topk(topk)
    
    return (e.data.numpy().squeeze().tolist() for e in topk)


def main():
    args = parse_args()
    img_path = args.img_path
    gpu = args.gpu
    topk = args.topk
    checkpoint = args.checkpoint
    
    print('='*10+'Params'+'='*10)
    print('Image path:       {}'.format(img_path))
    print('Load model from:  {}'.format(checkpoint))
    print('GPU mode:         {}'.format(gpu))
    print('TopK:             {}'.format(topk))
    
    # Load the model
    model, __, __ = load_checkpoint(checkpoint)
    class_names = model.class_names
    
    # Set the GPU
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print('Current device: {}'.format(device))
    model.to(device)
    
    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    # Predict
    print('='*10+'Predict'+'='*10)
    probs, classes = predict(img_path, model, device, topk)
    flower_names = [cat_to_name[class_names[e]] for e in classes]
    for prob, flower_name in zip(probs, flower_names):
        print('{:20}: {:.4f}'.format(flower_name, prob))
    
    
if __name__ == '__main__':
    main()