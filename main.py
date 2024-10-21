import os
from argparse import ArgumentError

import numpy as np
import torch
import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as tt

from NeuralNetwork.network import AnimalNetwork
from dataset.dataset import AnimalDataset
from dataset.load import load_labels
from dataset.transforms import load_transform, train_augmenter, validation_load_transform

from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True

def dump_to_cvs(file_names, predictions):
    if len(file_names) != len(predictions):
        raise ArgumentError(argument=None, message="predictions length does not match file names length")
    with open("result/result.csv", "w") as file:
        file.write("image_name,predicted_class\n")
        for i in range(len(file_names)):
            file.write(f"{file_names[i]},{predictions[i]}\n")


classes = {
    0: "Заяц", 1: "Кабан", 2: "Кошки", 3: "Куньи", 4: "Медведь", 5: "Оленевые", 6: "Пантеры", 7: "Полорогие",
    8: "Собачие", 9: 'Сурок'
}

labels = load_labels("train.csv")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = AnimalNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch in tqdm.tqdm(dataloader):
        X, y = batch
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


"""
        if batch % 5 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")"""

best_val_f1 = 0.0
best_model_path = 'best_model.pth'


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    global best_val_f1
    global best_model_path

    val_f1 = correct

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved with F1: {best_val_f1:.4f}')

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


IS_LEARNING = True

if IS_LEARNING:
    epochs = 20

    #model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    for t in range(epochs):

        train_keys, test_keys = train_test_split(list(labels.keys()), test_size=0.3, random_state=1)

        train_labels = [(label, labels[label]) for label in train_keys]
        test_labels = [(label, labels[label]) for label in test_keys]

        train_dataset = AnimalDataset("train", train_labels, load_transform, train_augmenter)
        test_dataset = AnimalDataset("train", test_labels, validation_load_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8, pin_memory=False, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, num_workers=8, pin_memory=False, shuffle=True)

        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
else:
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

infer_transform = tt.Compose([
    tt.Resize((512, 512)),
    tt.ToTensor(),
    tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

with torch.no_grad():
    model.eval()
    _, _, file_names = list(os.walk("test"))[0]
    res = []

    for file_name in tqdm.tqdm(file_names):
        full_path = "test" + "/" + file_name
        image = Image.open(full_path).convert('RGB')
        image = infer_transform(image)
        image.unsqueeze_(0)
        image = image.to(device)
        prediction = model(image)
        res.append(torch.argmax(model(image), dim=1).cpu().numpy()[0])

    plt.figure(figsize=(25, 25))
    for i in range(25):
        file_name = file_names[i]
        full_path = "test" + "/" + file_name
        image = Image.open(full_path).convert('RGB')
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        plt.title(classes[res[i]])
        plt.axis("off")

    plt.show()

    dump_to_cvs(file_names, res)