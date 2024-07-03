# Simple example of loading CIFAR-10 and SVHN datasets and
# generate their statistic meta-features for OOD detection
# import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import meta_feature.stats_mf
import meta_feature.earth_mover

# Define the transformations for the datasets
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 test dataset
cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create test dataloader for CIFAR-10
cifar10_testloader = DataLoader(cifar10_test, batch_size=len(cifar10_test), shuffle=False, num_workers=2)

# Load SVHN dataset
svhn_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)

# Create test dataloader for SVHN
svhn_testloader = DataLoader(svhn_test, batch_size=len(svhn_test), shuffle=False, num_workers=2)

# Get statistic meta-features for the dataset pair CIFAR10-SVHN

meta_vec = []
meta_vec_name = []

# cifar10 stats meta feature
stat_mf_cifar10, mf_name_cifar10 = meta_feature.stats_mf.extract_meta_features(cifar10_testloader) # data loader is passed in
meta_vec.extend(stat_mf_cifar10)
meta_vec_name.extend(mf_name_cifar10)

# EMD
emd_value = meta_feature.earth_mover.run(cifar10_testloader, svhn_testloader)
meta_vec.append(emd_value)
meta_vec_name.append('EMD')

# svhn stats meta feature
stat_mf_svhn, mf_name_svhn = meta_feature.stats_mf.extract_meta_features(svhn_testloader) # data loader is passed in
meta_vec.extend(stat_mf_svhn)
meta_vec_name.extend(mf_name_svhn)

print(len(meta_vec))

