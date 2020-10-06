# test backbone from torchvision
import torch
import torchvision
net = torchvision.models.resnet18()
print(torch.cuda.is_available())
print("test so h")