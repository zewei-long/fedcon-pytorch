# FedCon: A Contrastive Framework for Federated Semi-Supervised Learning
A PyTorch implementation for the paper **FedCon: A Contrastive Framework for Federated Semi-Supervised Learning**.

We have 5 baselines (FedAvg-FixMatch, FedProx-FixMatch, FedAvg-UDA, FedProx-UDA, FedMatch, SSFL) and our proposed FedCon framework in our experiment.

We do our experiments on MNIST, CIFAR-10, and SVHN datasets.

you should place your data in `./fedcon-ecmlpkdd-master/data/mnist` (mnist for example)





## Getting Started

python>=3.6
pytorch>=0.4

To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally).



## Some Exampless

We provide some examples here.



#### MNIST IID

> python fedcon-main.py --data_dir ../data/mnist --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --label_rate 0.01 --iid iid

#### MNIST IID & gamma (label_rate)=0.1

> python fedcon-main.py --data_dir ../data/mnist --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --label_rate 0.1 --iid iid

#### MNIST non-IID

> python fedcon-main.py --data_dir ../data/mnist --backbone Mnist --dataset mnist --batch_size 10 --num_epochs 200 --label_rate 0.01 --iid noniid

#### CIFAR-10 IID

> python fedcon-main.py --data_dir ../data/cifar --backbone Cifar --dataset cifar10 --batch_size 10 --num_epochs 200 --label_rate 0.01 --iid iid

#### SVHN IID

> python fedcon-main.py --data_dir ../data/svhn --backbone Svhn --dataset svhn --batch_size 10 --num_epochs 150 --label_rate 0.01 --iid iid



