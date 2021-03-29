import sys, humanize, psutil, GPUtil, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from .model import DeviceDataLoader

def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

#Using a GPU if available
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def load_set(param, device, dataset):
    path, shuffle_, batch_size = [value for key, value in param.items()]
    tfms = tt.Compose([tt.ToTensor()])
    ds = ImageFolder(dataset+path, tfms)
    dl = DataLoader(ds, batch_size, shuffle = shuffle_, num_workers=8, pin_memory=True)
    device_dl = DeviceDataLoader(dl, device)

    return device_dl

def predict_image(img, model, classMap, device):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return classMap[preds[0].item()]