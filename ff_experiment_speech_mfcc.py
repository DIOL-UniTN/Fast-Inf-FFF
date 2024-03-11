import os
import torch
import typer
import pickle
import mlflow
import numpy as np
import pandas as pd
from tqdm import trange
from fff_trainer import FF, train, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class SpeechDataset(Dataset):
    def __init__(self, fold="train"):
        df = pd.read_csv(f"data/speech_mfcc/speech_commands_preprocessed_mfcc/sa_{fold}.csv")
        self.data = []
        self.labels = []
        for _, (dir, name, label) in df.iterrows():
            self.data.append(np.load(f"data/speech_mfcc/speech_commands_preprocessed_mfcc/{dir}/{name.replace('wav', 'npy')}"))
            self.labels.append(label)
        self.data = np.array(self.data).astype(np.float32)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


def load_data():
    trainset = SpeechDataset("train")
    testset = SpeechDataset("test")

    # Select class to keep 
    trainloader = DataLoader(trainset, batch_size=512, shuffle=True)
    testloader = DataLoader(testset, batch_size=512)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples


def main(layer_width: int, epochs: int):
    trainloader, testloader, _ = load_data()
    net = FF(13*61, layer_width, 10).to(DEVICE)

    with mlflow.start_run(experiment_id="23"):
        mlflow.log_param("leaf_width", layer_width)
        mlflow.log_param("depth", 1)
        mlflow.log_param("epochs", epochs)

        # Train the net and log on mlflow
        for i in trange(epochs):
            train(net, trainloader, 1, norm_weight=0, lr=1e-3, weight_decay=5e-5)
            train_loss, train_acc = test(net, trainloader)
            test_loss, test_acc = test(net, testloader)

            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)

        # Log model
        mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    typer.run(main)
