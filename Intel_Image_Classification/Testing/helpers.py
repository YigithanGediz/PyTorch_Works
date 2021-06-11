import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
import os
from PIL import Image
def fit(model, optimizer, train_loader, test_loader,criterion, n_epochs, device, n_iterations, save_dir, vol=3):
    losses = list()
    acc = list()
    for epoch in range(1, n_epochs+1):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device).type(torch.long)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"Epoch {epoch}, iteration {i}, loss {loss}")
                losses.append(loss)

            if i == n_iterations/2 or i==n_iterations -1:
                accuracy = predict_over_set(test_loader, model, device)
                max_accuracy = max(acc) if len(acc) > 0 else 0
                print(f"Epoch {epoch + 1}, loss: {loss}, accuracy {accuracy}")
                acc.append(accuracy)
                if (accuracy) >= max_accuracy and accuracy >= 80:
                    print(f"{accuracy} is higher than the max {max_accuracy}")
                    torch.save(model.state_dict(),
                               os.path.join(save_dir, f"resnet{vol}_epoch_{epoch}_acc_{int(accuracy)}.pth"))

    return model, losses, acc



def predict_over_set(test_loader, model, device):
    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for inputs, labels in test_loader:
            n_samples += inputs.shape[0]

            inputs = inputs.to(device)
            labels = torch.flatten(labels.to(device)).type(torch.int32)

            logits = model(inputs)
            outputs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(outputs, dim=1).type(torch.int32)

            equals = torch.eq(predictions, labels).sum().item()

            n_correct += equals

    return (n_correct *100)/n_samples


def get_images(path_to_folders, imsize=(150,150)):
    folders = sorted(os.listdir(path_to_folders))
    image_arr = list()
    label_arr = list()
    label_decoders = {}
    for i,foldername in enumerate(folders):
        label_decoders[i] = foldername
        path = os.path.join(path_to_folders, foldername)
        images = os.listdir(path)

        for imname in images:
            image_path = os.path.join(path, imname)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, dsize=imsize, interpolation=cv2.INTER_NEAREST)
            image_arr.append(image)
            label_arr.append(i)

    return np.array(image_arr), np.array(label_arr), label_decoders


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform = None):
        self.x = x
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]
        self.transform = transform

    def __getitem__(self, item):
        image, label = self.x[item], self.y[item]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        sample = [image, label.type(torch.long)]
        sample[1] = sample[1].type(torch.long)

        return tuple(sample)

    def __len__(self):
        return self.n_samples

