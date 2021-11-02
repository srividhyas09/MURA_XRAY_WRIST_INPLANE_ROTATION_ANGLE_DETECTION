import json
import imutils
from skimage.io import imread
from skimage.color import rgb2gray
import pandas as pd
import cv2
from math import *
from PIL import Image, ImageEnhance
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class Preprocessing(Dataset):

    """ Dataset for XR wrist images"""
    def __init__(self, data_path, transform=None):
        """
        :param roo_dir: path to MURA-v1.1 directory.
        :param transform: Optional transform to b3e appied on a sample.
        """
        self.data_path = data_path
        self.transform = transform

        self.label_paths = pd.read_csv(data_path)

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        # Load labels
        image_path = self.label_paths['image_path'][idx]
        image = imread(image_path)

        if len(image.shape) > 2:
            image = rgb2gray(image) * 255
            image = image.astype(np.uint8)

        with open(image_path.replace(".png", ".json")) as label_file:
            label = json.load(label_file)

        radius = None
        for shape in label['shapes']:
            if shape['label'] == 'Radius':
                radius = shape['points']

        contour = [np.ceil(np.array(radius)).astype(np.int32)]
        mask = np.zeros(image.shape, np.uint8)
        mask = cv2.fillPoly(mask, pts=contour, color=(1))

        # Return sample
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Crop(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if image.dtype == np.float64:
            image = image * 255
            image = image.astype(np.uint8)

        blur = cv2.GaussianBlur(image, (3, 3), 0)

        edges = cv2.Canny(blur, 10, 50, 3)
        cnts, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cont_p = np.squeeze(np.vstack(cnts))
        mini = np.min(cont_p, axis=0)
        maxi = np.max(cont_p, axis=0)
        h, w = maxi - mini
        cropped = image[mini[1]:mini[1] + w, mini[0]:mini[0] + h]
        crop2 = mask[mini[1]:mini[1] + w, mini[0]:mini[0] + h]
        return {'image': cropped, 'mask': crop2}


class Resize(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        x, y = image.shape[:2]

        axis = np.argmin([x, y])

        pad_width = [(0, 0), (0, 0)]
        if (x - y) % 2 == 0:
            pad_width[axis] = (int(np.abs(x - y) / 2), int(np.abs(x - y) / 2))

        else:
            pad_width[axis] = (floor(np.abs(x - y) / 2), ceil(np.abs(x - y) / 2))

        img = np.pad(image, pad_width, 'constant', constant_values=0)
        mask = np.pad(mask, pad_width, 'constant', constant_values=0)
        return {'image': img, 'mask': mask}


class RandomRotate(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        angle = np.random.randint(-90, 90)

        if image is None:
            print(image)
        image = imutils.rotate(np.array(image), angle)
        mask = imutils.rotate(np.array(mask), angle)

        return {'image': image, 'mask': mask}


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        mask = cv2.resize(mask, (new_w, new_h))

        #img = img / np.max(img)
        #mask = mask / np.max(mask)
        return {'image': img, 'mask': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # if image has no grayscale color channel, add one
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if (len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        if (len(mask.shape) == 2):
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask = mask.transpose((2, 0, 1))


        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}


class Normalization(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        mean = image.mean()

        var = image.var()
        image = (image-mean) / sqrt(var)

        return {'image': image, 'mask': mask}
