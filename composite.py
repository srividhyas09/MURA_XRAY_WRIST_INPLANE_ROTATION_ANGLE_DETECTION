import time
import copy

# torch libs imports
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# imports from local
from iiml_mura.Research.Regression.Preprocessing import *
import matplotlib.pyplot as plt
from unet import UNet
from iiml_mura.Research.Regression.models.facedetection_CNN import Facedetection_CNN

def train_model(model, dataset, criterion, device, num_epochs=3, is_inception=False):
    since = time.time()

    test_loss = []
    val_loss = []
    best_model_wts = None
    batchsize=12

    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.7),
                                                                   len(dataset) - int(len(dataset) * 0.7)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=4)

    conv1x1 = nn.Conv2d(512, 200, (1, 1)).to(device)
    avg_pool = nn.AvgPool2d((3, 3)).to(device)

    fc1 = nn.Linear(5000, 150).to(device)
    fc1_drop = nn.Dropout(p=0.4).to(device)
    fc2 = nn.Linear(150, 2).to(device)

    p_list = list(fc1.parameters()) + list(fc2.parameters()) + list(conv1x1.parameters()) + list(model.parameters())
    optimizer = optim.Adam(p_list, lr=learning_rate)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_loader
            else:
                model.eval()  # Set model to evaluate mode
                dataloaders = valid_loader

            running_loss = 0.0

            # Iterate over data.
            for batch_index, sample in enumerate(dataloaders):
                inputs = sample['image']
                sin_cos = sample['sin_cos'].view(-1, 2)

                inputs = inputs.type(torch.FloatTensor)
                sin_cos = sin_cos.type(torch.FloatTensor)

                inputs = inputs.to(device)
                sin_cos = sin_cos.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    #outputs = model(inputs)

                    x1 = model.inc(inputs)
                    x2 = model.down1(x1)
                    x3 = model.down2(x2)
                    x4 = model.down3(x3)
                    x = model.down4(x4)

                    x = avg_pool(x)
                    x = conv1x1(x)

                    x = x.view(x.size(0), -1)

                    x = fc1(x)
                    x = torch.relu(x)
                    x = fc1_drop(x)
                    x = fc2(x)
                    
                    loss = criterion(x, sin_cos)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # ...log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch

            epoch_loss = running_loss / len(dataloaders.dataset)
            if phase == 'train':
                test_loss.append(epoch_loss)
            else:
                if val_loss != []:
                    if val_loss[-1] >= epoch_loss:
                        fc1_w = copy.deepcopy(fc1.state_dict())
                        fc2_w = copy.deepcopy(fc2.state_dict())
                        con1x1 = copy.deepcopy(conv1x1.state_dict())
                        mod = copy.deepcopy(model.state_dict())

                val_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


    time_elapsed = time.time() - since
    torch.save(fc1_w, 'pooling_CONV_fc1.pt')
    torch.save(fc2_w, 'pooling_CONV_fc2.pt')
    torch.save(con1x1, 'pooling_CONV_con1x1.pt')
    torch.save(mod, 'Umodel.pt')

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    plt.plot(test_loss)
    plt.title("AVG pooling_CONV_test")
    plt.show()
    plt.plot(val_loss)
    plt.title("AVG pooling_CONV_valid")
    plt.show()
    # load best model weights
    #model.load_state_dict()
    return best_model_wts


if __name__ == '__main__':
    data_transform_augment = transforms.Compose([Crop(), Resize(), RandomRotate(), Rescale((250, 250)), ToTensor()])
    dataset = Preprocessing('../Ground_truth.csv', transform=data_transform_augment)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = model.to(device)

    model_path = 'UnetBCE_normalized.pt'
    model.load_state_dict(torch.load(model_path))
    '''for param in model.parameters():
        param.requires_grad = False'''

    learning_rate = 0.0002
    criterion = nn.MSELoss()
    epochs = 100
    trained_model = train_model(model, dataset, criterion, device, num_epochs=epochs, is_inception=False)

    # after training, save your model parameters in the dir 'saved_models'
