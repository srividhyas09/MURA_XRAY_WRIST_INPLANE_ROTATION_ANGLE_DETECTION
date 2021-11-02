import torch
from torchvision import transforms
import torch.nn as nn
import math
import numpy as np
from iiml_mura.Research.Regression.Preprocessing import *
import matplotlib.pyplot as plt
from unet import UNet
from iiml_mura.Research.Regression.models.facedetection_CNN import Facedetection_CNN

#from models.resnet import *

if __name__ == '__main__':
    # load sample image
    data_transform_augment = transforms.Compose([Crop(), Resize(), Rescale((250, 250)), ToTensor()])
    dataset = Preprocessing('../Test.csv', transform=data_transform_augment)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = 'Umodel.pt'
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    criterion = nn.MSELoss()
    angle_error = []
    pred = []
    orig = []
    fc1_w = "pooling_CONV_fc1.pt"
    fc2_w = "pooling_CONV_fc2.pt"
    con1x1 = "pooling_CONV_con1x1.pt"

    conv1x1 = nn.Conv2d(512, 200, (1, 1))
    avg_pool = nn.AvgPool2d((3, 3))

    fc1 = nn.Linear(5000, 150)
    fc1_drop = nn.Dropout(p=0.3).to(device)
    fc2 = nn.Linear(150, 2)

    fc1.load_state_dict(torch.load(fc1_w))
    fc2.load_state_dict(torch.load(fc2_w))
    conv1x1.load_state_dict(torch.load(con1x1))

    for batch_index, sample in enumerate(dataloader):
        inputs = sample['image']

        image = inputs.type(torch.FloatTensor)

        x1 = model.inc(image)
        x2 = model.down1(x1)
        x3 = model.down2(x2)
        x4 = model.down3(x3)
        x = model.down4(x4)

        x = conv1x1(x)
        x = avg_pool(x)

        x = x.view(x.size(0), -1)

        x = fc1(x)
        x = torch.relu(x)
        x = fc1_drop(x)

        preds = fc2(x)


        if abs(preds[0][0]) <= 1:
            prediction = math.degrees(math.asin(preds[0][0]))
            actual = math.degrees(math.asin(sample['sin_cos'][0][0]))
            angle_error.append(prediction - actual)
            pred.append(prediction)
            orig.append(actual)
        else:
            print("error for:", preds, sample['sin_cos'])
        '''plt.imshow(inputs.squeeze())
        plt.title("Faceorig"+str(actual)+ " " + str(prediction))
        plt.show()'''
    angle_error = np.array(angle_error)
    print(np.mean(np.abs(angle_error)), np.median(np.abs(angle_error)), np.min(np.abs(angle_error)), np.max(np.abs(angle_error)))
    print(len(orig), len(pred))
    plt.scatter(orig, pred)
    plt.plot(orig, orig, linestyle='solid')
    plt.xlabel("original")
    plt.ylabel("predictions")
    plt.title("Full composite ntw trained")

    #plt.scatter(orig, angle_error)
    plt.show()

