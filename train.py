import time
import copy

# torch libs imports
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# imports from local
from Seg_preprocessing import *
import matplotlib.pyplot as plt
from unet import UNet

def train_model(model, dataset, criterion, optimizer, device, num_epochs=3, is_inception=False):
    since = time.time()

    test_loss = []
    val_loss = []
    best_model_wts = None
    batchsize=4

    train_data, val_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.7),
                                                                   len(dataset) - int(len(dataset) * 0.7)])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=True, num_workers=4)

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
                mask = sample['mask']

                inputs = inputs.type(torch.FloatTensor)
                mask = mask.type(torch.FloatTensor)

                inputs = inputs.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, mask)


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
                        best_model_wts = copy.deepcopy(model.state_dict())
                val_loss.append(epoch_loss)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    plt.plot(test_loss)
    plt.show()
    plt.plot(val_loss)
    plt.show()
    # load best model weights
    #model.load_state_dict()
    return best_model_wts


if __name__ == '__main__':
    data_transform_augment = transforms.Compose([Crop(), Resize(), RandomRotate(), Rescale((250, 250)), Normalization(), ToTensor()])

    dataset = Preprocessing('../Ground_truth.csv', transform=data_transform_augment)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    #model = DinkNet34()
    model = model.to(device)

    learning_rate = 0.0002
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 100
    trained_model = train_model(model, dataset, criterion, optimizer, device, num_epochs=epochs, is_inception=False)

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(trained_model, 'UnetBCE_normalized.pt')
