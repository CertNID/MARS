import collections
from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

os.environ["/home/huan1932/data/ImageNet"] = "/home/huan1932/data/ImageNet"
common = '/home/huan1932/CertNID/CIC-IDS2018/CIC-IDS2018-ACID/normalized'
norm = "/home/huan1932/data/CIC-IDS2018/unnormalized/"
IMAGENET_LOC_ENV = "/home/huan1932/data/ImageNet"
cadedataset = "/home/huan1932/OtherCAR/BARS/smoothed_cade/data"
# list of all datasets
DATASETS = ['ImageNet','ids18','newhulk','newinfiltration','ACID_unnormalized.npz',"ids18_unnormalized"]




            
# def _cade(dataset,split: str="train"):
#     data_dir = ''
#     if dataset == "newhulk":
#         data_dir = os.path.join(cadedataset,dataset)
#     elif dataset == 'newinfiltration':
#         data_dir = os.path.join(cadedataset,dataset)
#     if split == "train":
#         x_train = np.load(os.path.join(data_dir, "X_train.npy"))
#         y_train= np.load(os.path.join(data_dir, "y_train.npy"))

#         class_map, num_classes_train = get_class_map(data_dir)
        
#         y_train_class_new = np.zeros_like(y_train, dtype=np.int64)
        
#         for k, v in class_map.items():
#             y_train_class_new[y_train == k] = v

#         return torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train_class_new, dtype=torch.long), num_classes_train, class_map

#     elif split == "test":
#         x_test = np.load(os.path.join(data_dir, "X_test.npy"))
#         y_test = np.load(os.path.join(data_dir, "y_test.npy"))

#         class_map, num_classes_test = get_class_map(data_dir)
       
#         y_test_class_new = np.zeros_like(y_test, dtype=np.int64)

#         for k, v in class_map.items():
#             y_test_class_new[y_test == k] = v

#         y_train_class = np.load(os.path.join(data_dir, "y_train.npy"))
#         train_class_set = np.unique(y_train_class).tolist()
        
#         y_test_drift = np.ones_like(y_test, dtype=np.int64)

#         for i in range(len(train_class_set)):
#             y_test_drift[y_test == train_class_set[i]] = 0

#         return torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test_class_new, dtype=torch.long), num_classes_test, class_map, torch.tensor(y_test_drift, dtype=torch.long)

#     else:
#         raise NotImplementedError()
    
def _cade(dataset,split: str="train"):
    if dataset == "newhulk":
        data_dir = '/home/huan1932/CertNID/CIC-IDS2018/IDS2018CADE/data/IDS_new_Hulk.npz'
    elif dataset == 'newinfiltration':
        data_dir = '/home/huan1932/CertNID/CIC-IDS2018/IDS2018CADE/data/IDS_new_Infilteration.npz'
    if split == "train":
        data = np.load(data_dir)
        
        x_train =data["X_train"]
        y_train= data["y_train"]

        class_map, num_classes_train = get_class_map(data_dir)
        
        y_train_class_new = np.zeros_like(y_train, dtype=np.int64)
        
        for k, v in class_map.items():
            y_train_class_new[y_train == k] = v

        return torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train_class_new, dtype=torch.long), num_classes_train, class_map

    elif split == "test":
        data = np.load(data_dir)
        
        x_test =data["X_test"]
        y_test= data["y_test"]

        class_map, num_classes_test = get_class_map(data_dir)
       
        y_test_class_new = np.zeros_like(y_test, dtype=np.int64)

        for k, v in class_map.items():
            y_test_class_new[y_test == k] = v

        y_train_class = data["y_train"]
        train_class_set = np.unique(y_train_class).tolist()
        
        y_test_drift = np.ones_like(y_test, dtype=np.int64)

        for i in range(len(train_class_set)):
            y_test_drift[y_test == train_class_set[i]] = 0

        return torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test_class_new, dtype=torch.long), num_classes_test, class_map, torch.tensor(y_test_drift, dtype=torch.long)

    else:
        raise NotImplementedError() 
    
# def get_class_map(data_dir: str):
#     y_train = np.load(os.path.join(data_dir, "y_train.npy"))
#     class_train = np.unique(y_train).tolist()
#     class_train.sort()
#     y_test = np.load(os.path.join(data_dir, "y_test.npy"))
#     class_test = np.unique(y_test).tolist()
    # class_test.sort()
    # class_map = collections.OrderedDict()

    # for i in range(len(class_train)):
    #     class_map.update({class_train[i]: len(class_map)})
    # for i in range(len(class_test)):
    #     if class_test[i] not in class_train:
    #         class_map.update({class_test[i]: len(class_map)})
        
    # return class_map, len(class_train)
def get_class_map(data_dir: str):
    y_train = np.load(data_dir)["y_train"]
    class_train = np.unique(y_train).tolist()
    class_train.sort()
    y_test = np.load(data_dir)["y_test"]
    class_test = np.unique(y_test).tolist()
    class_test.sort()
    class_map = collections.OrderedDict()

    for i in range(len(class_train)):
        class_map.update({class_train[i]: len(class_map)})
    for i in range(len(class_test)):
        if class_test[i] not in class_train:
            class_map.update({class_test[i]: len(class_map)})
        
    return class_map, len(class_train)

def get_dataset(dataset:str,split: str) -> Dataset:
    if dataset == "ImageNet":
        return _imagenet(split)
    elif dataset == "ids18":
        return _2018(split)
    elif dataset == "ids18_unnormalized":
        return _2018n(split)
    elif dataset == 'newinfiltration' or dataset == 'newhulk':
        return _cade(dataset,split)
    


def get_num_classes(dataset:str):
    if dataset == "ImageNet":
        return 1000
    elif dataset == 'ids18' or dataset == 'ids18_unnormalized':
        return 4
    elif dataset == 'newhulk' or dataset == 'newinfiltration'or dataset == 'newhulk_unnormalized'or dataset == 'newinfiltration_unnormalized':
        return 2
    


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "ImageNet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)



_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]


def _2018(split:str):
    # data_dir = os.path.join(common, DATASETS[1])
    data_dir = common
    if split == 'train':

        x_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        return [torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]

    elif split == "test":
        x_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))

        return [torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
        
def _2018n(split:str):
    data_dir = os.path.join(norm, DATASETS[4])
    x = np.load(data_dir)
    keys = x.files
    if split == 'train':
        x_train = x[keys[0]]
        result = np.delete(x_train,3,axis=1)
        y_train = x[keys[1]]
        y_train[y_train == 4] = 2
        y_train[y_train == 6] = 3
        return [torch.tensor(result, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)]

    elif split == "test":
        x_test = x[keys[2]]
        result = np.delete(x_test,3,axis=1)
        y_test = x[keys[3]]
        y_test[y_test == 4] = 2
        y_test[y_test == 6] = 3
        return [torch.tensor(result, dtype=torch.float), torch.tensor(y_test, dtype=torch.long)]
        


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = IMAGENET_LOC_ENV
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


class NormalizeLayer(torch.nn.Module):
    

    def __init__(self, means: List[float], sds: List[float]):
        
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds
