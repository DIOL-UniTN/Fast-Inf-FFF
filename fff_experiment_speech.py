import os
import torch
import typer
import pickle
import mlflow
import numpy as np
import pandas as pd
from tqdm import trange
from fff_trainer import Net, train, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


# class SpeechDataset(Dataset):
#     def __init__(self, fold):
#         self.feat_path = "data/speech_commands_preprocessed/"
#         self.csv_path = f"data/speech_commands_preprocessed/sa_{fold}.csv"
#         df = pd.read_csv(self.csv_path)
#         self.info_list = df.values.tolist()
#         del df

#     def __len__(self):
#         return len(self.info_list)

#     def __getitem__(self, item):
#         folder, name, label = self.info_list[item]
#         feat_loc = os.path.join(self.feat_path, '\\' + str(folder), '{}.npy'.format(name.split(".")[0]))
#         feat = np.load(feat_loc)
#         return feat, label

class SpeechDataset(Dataset):
    def __init__(self, fold="train"):
        df = pd.read_csv(f"data/speech_commands_preprocessed/sa_{fold}.csv")
        self.data = []
        self.labels = []
        for _, (dir, name, label) in df.iterrows():
            self.data.append(np.load(f"data/speech_commands_preprocessed/{dir}/{name.replace('wav', 'npy')}"))
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
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    testloader = DataLoader(testset, batch_size=1024)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples


def main(leaf_width: int, depth: int, epochs: int, norm_weight: float):
    trainloader, testloader, _ = load_data()
    net = Net(65*65, leaf_width, 10, depth, 0, 0).to(DEVICE)

    with mlflow.start_run(experiment_id="4"):
        mlflow.log_param("leaf_width", leaf_width)
        mlflow.log_param("depth", depth)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("norm_weight", norm_weight)
        mlflow.log_param("hardened", net.fff.train_hardened)

        # Train the net and log on mlflow
        for i in trange(epochs):
            train(net, trainloader, 1, norm_weight=norm_weight)
            train_loss, train_acc = test(net, trainloader)
            test_loss, test_acc = test(net, testloader)

            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)

        # Evaluation
        net.eval()
        train_loss, train_acc = test(net, trainloader)
        test_loss, test_acc = test(net, testloader)
        mlflow.log_metric("eval_train_accuracy", train_acc)
        mlflow.log_metric("eval_train_loss", train_loss)
        mlflow.log_metric("eval_test_accuracy", test_acc)
        mlflow.log_metric("eval_test_loss", test_loss)

        # Log model
        mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    typer.run(main)
