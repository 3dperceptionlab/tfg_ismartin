# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

#apt install wget
import torch    #pip install torch
from torch.autograd import Variable as V
import torchvision.models as models #pip install torchvision
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image

def predictStage(img):
    #img = Image.open('../interactiveObjectsYolo/images/bedroom.jpg')
    #img = Image.open('../interactiveObjectsYolo/images/frame.jpg')
    input_img = V(centre_crop(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))