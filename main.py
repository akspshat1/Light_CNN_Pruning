import json
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CelebA
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18

from trainer import Trainer
from models.weak_cnn import WeakCNN
from models.light_weight_cnn import LightweightCNN

def load_config_from_json(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

config = load_config_from_json('config.json')

cifar_transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
transforms.Resize((32, 32))
])
train_dataset = CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
val_dataset = CIFAR10(root="./data", train=False, download=True, transform=cifar_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

print(f"Model: {config['model']}, Algorithm: {config['algorithm']}, Data Augmentation: {config['data_augmentation']}, Pretrained: {config['pretrained']}, Pruning: {config['pruning']}")

if config['model'] == 'simple':
    model = WeakCNN()
elif config['model'] == 'heavy':
    model = resnet18(num_classes = 10)
else:
    model = LightweightCNN()
    
trainer = Trainer(model, train_algorithm=config['algorithm'], is_data_augmentation=config['data_augmentation'], is_pretrained=config['pretrained'], is_pruning=config['pruning'], model_type=config['model'])
trainer.train(train_loader, val_loader)