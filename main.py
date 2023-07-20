import torch
import torch.nn.functional as F

from PIL import Image
import time


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tempfile import TemporaryDirectory


import torchvision
from torchvision import models

from torchvision import transforms as T

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


from dataset import ImageFolder




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    #with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join("/home/rabink1/IDL_workshop/IDL/proj", 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).double()
                labels = labels.to(device).long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'eval' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))
    return model



if __name==__main__:
    data_transforms = {
    'train': T.Compose([
        #T.ToTensor()
    ]),
    'eval': T.Compose([
        #T.ToTensor()

    ]),}

    data_dir = "/home/rabink1/D1/vtf_images/tuh"
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'eval']}

    dataloaders = {x: torch.utils.data.DataLoader(torch.utils.data.Subset(image_datasets[x],range(20)), 
                  batch_size=2,shuffle=True, num_workers=4) for x in ['train', 'eval']
                 }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_sizes = {x: len(torch.utils.data.Subset(image_datasets[x],range(20))) for x in ['train', 'eval']}

    res=models.video.r2plus1d_18(weights="KINETICS400_V1")
    res.stem[0]= torch.nn.Conv3d(1, 45, kernel_size=(1, 3, 3), stride=(1, 2, 2), bias=False)
    res.fc = torch.nn.Linear(in_features=512,out_features= 2)
    res.double()

    model_ft = res.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=2)

    model_ft.training
    model_ft.eval()
    model_ft.double()
    output=model_ft(inputs[0].double().unsqueeze(0))
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()

    #integrated_gradients = IntegratedGradients(model_ft)
    #attributions_ig = integrated_gradients.attribute(inputs[0].double().unsqueeze(0), 
            target=pred_label_idx, n_steps=100)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    attributions_ig_nt = noise_tunnel.attribute(inputs[0].double().unsqueeze(0), 
            nt_samples=5, nt_type='smoothgrad_sq', target=classes[0])

    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt[0,:,0,:,:].cpu().detach().numpy(),(1,2,0)),
                                      np.transpose(inputs[0,:,0,:,:].cpu().detach().numpy(),(1,2,0)),
                                      ["original_image", "heat_map"],
                                      ["all", "positive"],
                                      cmap=default_cmap,
                                      show_colorbar=True,
                                      fig_size=(20,8))
    np.save("noisy_ig_nt",np.transpose(attributions_ig_nt[0,:,0,:,:].cpu().detach().numpy(),(1,2,0)))




