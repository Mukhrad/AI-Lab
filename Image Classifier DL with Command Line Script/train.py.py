import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

import argparse

import time
import os
import copy


def parse_args():
    parser = argparse.ArgumentParser(description="Trains a network on a dataset of images and saves the model to a checkpoint")
    parser.add_argument('--data_dir', default='flowers', type=str, help='set thepath')
    parser.add_argument('--arch', default='resnet', type=str, help='the model architecture')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--hidden_layers', default=None, nargs='+', type=int, help='list of integers, the sizes of the hidden layers')
    parser.add_argument('--epochs', default=5, type=int, help='num of training epochs')
    parser.add_argument('--gpu', default=False, type=bool, help='set the gpu mode')
    parser.add_argument('--checkpoint', default='my_point.pth', type=str, help='path to save checkpoint')
    args = parser.parse_args()
    return args

    """
    Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html  
    """
def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device, num_epochs=2):
    

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model)
    
    return model


def load_classifier(in_features, hidden_layers, out_features):
    
    
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(in_features, out_features))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], out_features))
        
    return classifier


def main():
    args = parse_args()
    data_dir = args.data_dir
    gpu = args.gpu
    arch = args.arch
    lr = args.lr
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    checkpoint = args.checkpoint
    
    print('='*10+'Params'+'='*10)
    print('Data dir:      {}'.format(data_dir))
    print('Model:         {}'.format(arch))
    print('Hidden layers: {}'.format(hidden_layers))
    print('Learning rate: {}'.format(lr))
    print('Epochs:        {}'.format(epochs))

    # Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=25,
                                                 shuffle=True, num_workers=1)
                  for x in ['train', 'valid', 'test']}

    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes
    
    # Set the GPU
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current device: {}'.format(device))
    
        
    # Choose the pretrained model
    if arch == 'resnet':
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
    else:
        print("Unknown model, please choose 'resnet' or 'vgg'")
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Redesign the classifier
    classifier = load_classifier(in_features, hidden_layers, 102)
    
    if arch == 'resnet':
        model.fc = classifier
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    elif arch == 'vgg':
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    else:
        pass
        
    print('='*10 + ' Architecture ' + '='*10)
    print('The classifier architecture:')
    print(classifier)
    
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    
    ## Train
    print('='*10 + ' Train ' + '='*10)
    model = train_model(dataloaders, dataset_sizes, model, criterion, optimizer, device, epochs)
    
    ## Test
    print('='*10 + ' Test ' + '='*10)
    model.eval()

    accuracy = 0
    
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Class with the highest probability is our predicted class
        equality = (labels.data == outputs.max(1)[1])
    
        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    print("Test accuracy: {:.3f}".format(accuracy/len(dataloaders['test'])))
    
    ## Save the checkpoint
    print('='*10 + ' Save ' + '='*10)
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.class_names = class_names
    
    checkpoint_dict = {'epoch': epochs,
                  'model': model,
                  'optimizer': optimizer.state_dict()}
    
    torch.save(checkpoint_dict, checkpoint)
    print('Save the trained model to {}'.format(checkpoint))
    
    
if __name__ == '__main__':
    main()