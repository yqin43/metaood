import numpy as np
import time

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models import resnet50

from pytorch_ood.detector import MaxSoftmax
from pytorch_ood.model import WideResNet
from pytorch_ood.utils import OODMetrics, ToUnknown

from PIL import Image
import os
from scipy.stats import skew, kurtosis


in_data_lst = ['cifar10','cifar100','imagenet','fashionmnist']
ood_data_lst = ['cifar10','cifar100','mnist','places365','svhn','texture','tin','ssb_hard','ninco','inaturalist','textures','openimage_o']

directory_list_0 = [
            'data/benchmark_imglist/cifar10/test_cifar10.txt',
            'data/benchmark_imglist/cifar100/test_cifar100.txt',
            'data/benchmark_imglist/mnist/test_mnist.txt',
            'data/benchmark_imglist/cifar100/test_places365.txt',
            'data/benchmark_imglist/cifar100/test_svhn.txt',
            'data/benchmark_imglist/cifar100/test_texture.txt',
            'data/benchmark_imglist/cifar100/test_tin.txt'
        ]

directory_list_1 = [
    "data/benchmark_imglist/imagenet200/test_ssb_hard.txt",
    "data/benchmark_imglist/imagenet200/test_ninco.txt",
    "data/benchmark_imglist/imagenet200/test_inaturalist.txt",
    "data/benchmark_imglist/imagenet200/test_textures.txt",
    "data/benchmark_imglist/imagenet200/test_openimage_o.txt"
]

class CustomDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None, target_transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with image paths and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # Read the txt file
        self.img_labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                self.img_labels.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        full_img_path = os.path.join(self.img_dir, img_path)
        image = Image.open(full_img_path).convert('RGB')  # Convert to RGB in case of grayscale images

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class ToRGB(object):
    """
    Convert Image to RGB, if it is not already.
    """

    def __call__(self, x):
        try:
            return x.convert("RGB")
        except Exception as e:
            return x

def landmarker(model, data, d2):
    landmarker = []
    dataset_out_test = d2
    dataset_in_test = data
    
    test_loader = DataLoader(dataset_in_test+dataset_out_test, batch_size=128)

    model = model.to(device)
    model = model.eval()

    detector  = MaxSoftmax(model)
    metrics = OODMetrics()

    for x, y in test_loader:
        metrics.update(detector(x.to(device)), y)
    scores = metrics.buffer.get("scores").view(-1).numpy()

    softmax_probs = scores
    entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10))
    top1_softmax_prob = np.max(softmax_probs)
    top2_softmax_prob = np.partition(softmax_probs, -2)[-2]
    softmax_prob_range = np.ptp(softmax_probs)
    confidence_margin = top1_softmax_prob - top2_softmax_prob

    landmarker.extend([entropy, top1_softmax_prob, top2_softmax_prob, softmax_prob_range, confidence_margin])
    landmarker.extend([np.mean(scores), np.std(scores), np.min(scores), np.max(scores)])
    skew_s, kurtosis_s = skew(scores.squeeze()), kurtosis(scores.squeeze())
    landmarker.extend([skew_s, kurtosis_s])

    return landmarker
    


big_vec = []
for d1 in in_data_lst:
    start_time = time.time()
    #load data
    if d1 == 'fashionmnist':
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # Convert grayscale to 3-channel
            transforms.Resize((32,32)), # Resize images to match ResNet input
            transforms.ToTensor(), # Convert images to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalization
            ])
        data = datasets.FashionMNIST(root='./data/FashionMNIST', train=False, transform=trans, download=True)
        train_d = datasets.FashionMNIST(root='./data/FashionMNIST', train=True, transform=trans, download=True)

        model = WideResNet(num_classes=10)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_loader = DataLoader(train_d, batch_size=128, shuffle=True)
        num_epochs = 99
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        model = model.eval().to(device)

    elif d1 == 'imagenet':
        print('imgnet loaded')
        trans = ResNet50_Weights.IMAGENET1K_V1.transforms()
        data = datasets.ImageNet('data/images_largescale/imagenet_1k/train1', split='val', transform=trans)
        train_d = datasets.ImageNet('data/images_largescale/imagenet_1k/train1', split='train', transform=trans)
        model = resnet50(ResNet50_Weights.IMAGENET1K_V1).eval().to(device)
        
    else: # cifar10 and cifar100
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        trans = transforms.Compose([
                transforms.Resize(size=(32, 32)),
                ToRGB(),
                transforms.ToTensor(),
                transforms.Normalize(std=std, mean=mean),
            ])
        img_dir = "data/images_classic"
        txt_dir = f'data/benchmark_imglist/{d1}/test_{d1}.txt'
        txt_d_dir = f'data/benchmark_imglist/{d1}/train_{d1}.txt'
        
        data = CustomDataset(txt_path=txt_dir, img_dir=img_dir, transform=trans)
        train_d = CustomDataset(txt_path=txt_d_dir, img_dir=img_dir, transform=trans)
        if d1 == 'cifar10':
            model = WideResNet(num_classes=10, pretrained=f'cifar10-pt')
        elif d1 == 'cifar100':
            model = WideResNet(num_classes=100, pretrained=f'cifar100-pt')

    # ood data
    test_out = []
    if d1 == 'cifar10':
        directory_list = directory_list_0.copy()
        del directory_list[0]
    elif d1 == 'cifar100':
        directory_list = directory_list_0.copy()
        del directory_list[1]
    else:
        directory_list = directory_list_0

    for txt_dir in directory_list:
        img_dir = "data/images_classic"
        test_out.append(
            CustomDataset(txt_path=txt_dir, img_dir=img_dir, transform=trans, target_transform=ToUnknown())
        )

    for txt_dir in directory_list_1:
        img_dir = "data/images_largescale"
        if 'textures' in txt_dir:
            img_dir = "data/images_classic"
        test_out.append(
            CustomDataset(txt_path=txt_dir, img_dir=img_dir, transform=trans, target_transform=ToUnknown())
        )
    
    print('ood data loaded')
    j = 0
    for d2 in test_out:
        j = j+1

        # msp landmarker
        l = landmarker(model,data,d2)
        big_vec.append(l)
        
        with open('full_landmarker_msp.npy', 'wb') as f:
            np.save(f, np.array(big_vec)) 

        print('saved',j, time.time()-start_time)