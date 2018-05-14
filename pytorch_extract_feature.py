# the example script to extract features using pyTorch CNN model
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import pdb
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
from dataset import Dataset
import torch.utils.data as data
import glob
import argparse
import gc
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

parser = argparse.ArgumentParser(description="Pytorch feature extraction")
parser.add_argument("--image_path", dest="image_path", type=str, required=True, help="path to directory containing images")
parser.add_argument("--layer_names", dest="layer_names", type=str, required=True,
                    help="layer names separated by commas")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=64, help="batch size (64)")
args = parser.parse_args()

torch.cuda.set_device(0)
# image datasest to be processed
name_dataset = 'imagenetval'
layer_names = args.layer_names.split(",")
# root_image = 'data/images'
# with open('data/images/imagelist.txt') as f:
#     lines = f.readlines()
# imglist = []
# for line in lines:
#     line = line.rstrip()
#     imglist.append(root_image + '/' + line)
imglist = glob.glob(args.image_path + "/*")
imglist.sort(key=natural_keys)

# load the pre-trained weights
name_model = 'vgg16'
# model_file = '/data/vision/oliva/scenedataset/places2new/models/whole_wideresnet18_places365.pth.tar'
model = models.vgg16(pretrained=True) #torch.load(model_file)
model.eval()
model.cuda()

# features_names = ['features']
#features_names = ['layer4','avgpool'] # this is the last conv layer and global average pooling layers



def get_torch_later_from_name(name):
    if name == 'ReLU':
        return torch.nn.modules.activation.ReLU
    elif name == 'Conv2d':
        return torch.nn.modules.Conv2d
    elif name == 'Linear':
        return torch.nn.modules.Linear




# dataset setup
img_size = (224, 224) # input image size
batch_size = args.batch_size
num_workers = 6

# image transformer
tf = trn.Compose([
        trn.Resize(img_size),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = Dataset(imglist, tf)

loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)




features_blobs = []
def hook_feature(module, input, output):
    # hook the feature extractor
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

for layer in layer_names:
    for idx, feat in enumerate(model._modules['features']):
        if type(feat) == get_torch_later_from_name(layer):
            print ("Proccessing layer %s index %d" % (layer, idx))
            hook = model._modules['features'][idx].register_forward_hook(hook_feature)

            # save variables
            imglist_results = []
            features_results = [None] * 1
            num_batches = len(dataset) / batch_size
            for batch_idx, (input, paths) in enumerate(loader):
                del features_blobs[:]
                print ('%d / %d' % (batch_idx, num_batches))
                input = input.cuda()
                with torch.no_grad():
                    # input_var = V(input, volatile=True)
                    input_var = V(input)
                    logit = model.forward(input_var)
                    imglist_results = imglist_results + list(paths)
                    if features_results[0] is None:
                        # initialize the feature variable
                        for i, feat_batch in enumerate(features_blobs):
                            size_features = ()
                            size_features = size_features + (len(dataset),)
                            size_features = size_features + feat_batch.shape[1:]
                            features_results[i] = np.zeros(size_features)
                            print (features_results[i].shape)
                    start_idx = batch_idx*batch_size
                    end_idx = min((batch_idx+1)*batch_size, len(dataset))
                    for i, feat_batch in enumerate(features_blobs):
                        features_results[i][start_idx:end_idx] = feat_batch

            # save the features
            save_name = name_dataset  + '_' + name_model + '_' + layer + '_' + idx
            np.savez('%s.npz'%save_name, features=features_results)
            del features_results
            gc.collect()
            hook.remove()






# save_matlab = 0
# if save_matlab == 1:
#     import scipy.io
#     scipy.io.savemat('%s.mat'%save_name, mdict={'list': imglist_results, 'features': features_results[0]})
