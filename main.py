# coding: utf-8

import timm
import torch
from torchvision import transforms
from PIL import Image
from pprint import pprint

pprint(timm.list_models('*swin*'))

model_swinv2 = timm.create_model('swinv2_base_window8_256', pretrained=True)

processor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

image1 = Image.open('2-191129140547.jpg')
image2 = Image.open('1532570308842645.jpg')
image3 = Image.open('15-211109164I9518.jpg')

t1 = processor(image1)
t2 = processor(image2)
t3 = processor(image3)
t1 = t1.resize(1, 3, 256, 256)
t2 = t2.resize(1, 3, 256, 256)
t3 = t3.resize(1, 3, 256, 256)

o1 = model_swinv2(t1)
o2 = model_swinv2(t2)
o3 = model_swinv2(t3)

t_sim1 = torch.cosine_similarity(o1, o2)
t_sim2 = torch.cosine_similarity(o1, o3)
a = 1

from timm.optim import create_optimizer_v2
from timm.loss import *
