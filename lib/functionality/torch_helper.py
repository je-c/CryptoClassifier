import sys, humanize, psutil, GPUtil, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class DeviceDataLoader():
    """
    DeviceDataLoader Class
    ----------------------
    Wraps and sends a pytorch dataloader to current device
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """
        Move dataloader to device and yield a single batched sample
        """
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """
        Number of batches
        """
        return len(self.dl)

def mem_report():
    """
    Returns available device and device properties
    """
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def get_default_device():
    """
    Return current default device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """
    Loads data onto default device
        * :param data(torch.tensor): Dataset to load
        * :param device(torch.device): Device to load to
    :return (torch.device): Data loaded onto default device
    """
    if isinstance(data, (list,tuple)): return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)

def load_set(param, device, dataset):
    """
    Loads a batch of data to the device
        * :param param(dict): Batch parameters
        * :param device(torch.device): Device to load to
        * :param dataset(torch.tensor): Data to load
    :return (DeviceDataLoader): Batch data loaded onto default device
    """
    path, shuffle_, batch_size = [value for key, value in param.items()]
    transforms = tt.Compose([tt.ToTensor()])
    ds = ImageFolder(dataset+path, tfms)
    dl = DataLoader(
            ds, 
            batch_size, 
            shuffle = shuffle_, 
            num_workers=8, 
            pin_memory=True
        )
    device_dl = DeviceDataLoader(dl, device)

    return device_dl

def predict_image(image, model, classMap, device):
    """
    Predicts the class of a single image
        * :param img(np.ndarray): Numpy array of pixel/channel values
        * :param model(torch.nn.module): Model
        * :param classMap(dict): Mapped class values for prediction output
        * :param device(torch.device): Device to load data onto
    :return (str): Class prediction for the image
    """
    X = to_device(image.unsqueeze(0), device)
    _, prediction  = torch.max(model(X), dim=1)
    
    return classMap[prediction[0].item()]