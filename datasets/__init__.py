import torch
import torchvision


def get_dataset(dataset, data_dir, transform, train=True, download=False, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
    elif dataset == 'svhn':
        if train == True:
            dataset = torchvision.datasets.SVHN(data_dir, split = 'train', transform=transform, download=download)
        else:
            dataset = torchvision.datasets.SVHN(data_dir, split = 'test', transform=transform, download=download)
    else:
        raise NotImplementedError
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch

    return dataset


    # ython main.py --model simsiam --optimizer sgd --data_dir ./data/cifar --output_dir ./outputs/ --backbone resnet18 --dataset cifar10 --batch_size 32 --num_epochs 2  --weight_decay 0.0005 --base_lr 0.03 --warmup_epochs 10