import os
import torch
import typer
import pickle
import mlflow
import numpy as np
import pandas as pd
from tqdm import trange
from fff_trainer import FF, train_kd, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class SpeechDataset(Dataset):
    def __init__(self, fold="train"):
        df = pd.read_csv(f"data/speech_mfcc/sa_{fold}.csv")
        self.data = []
        self.labels = []
        for _, (dir, name, label) in df.iterrows():
            self.data.append(np.load(f"data/speech_mfcc/{dir}/{name.replace('wav', 'npy')}"))
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


TEACHER_RUNIDS = {1024: "b18f83f31eab403a984c6247b4659b21", # layer_width: run_id
                  512: "e3158575fc32488fbf779f40b2918d7b", # Only this changed w/o softmax
                  256: "ed131afd277f472ca75c979047fd9429",
                  128: "f29edbfa5ad241e483f79ec6130673ce",
                  64: "c91c9e0fdd51486ca9e06ee858ed3d12",
                  32: "1aaaa43771b0475f8ea6bf7d97892d46",
                  # 2512: "4a83128b58954e30808b532781da778a", # version 2
                  }
def main(layer_width: int, epochs: int, teacher_layer_width:int, kd_alpha:float, temp:float):
    trainloader, testloader, _ = load_data()
    net = FF(13*61, layer_width, 10).to(DEVICE)
    mlflow.artifacts.download_artifacts(run_id=TEACHER_RUNIDS[teacher_layer_width], 
                                        artifact_path='model/data/model.pth', dst_path=".")
    teacher = torch.load('model/data/model.pth').to(DEVICE)
    # Teacher Evaluation
    teacher.eval()
    train_loss, train_acc = test(teacher, trainloader)
    test_loss, test_acc = test(teacher, testloader)

    with mlflow.start_run(experiment_id="22"):
        mlflow.log_param("leaf_width", layer_width)
        mlflow.log_param("depth", 1)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("teacher_layer_width", teacher_layer_width)
        mlflow.log_param("teacher_run_id", TEACHER_RUNIDS[teacher_layer_width])
        mlflow.log_param("kd_alpha", kd_alpha)
        mlflow.log_param("temperature", temp)

        # Train the net and log on mlflow
        for i in trange(epochs):
            train_kd(net, teacher, trainloader, 1, norm_weight=0, lr=1e-3, weight_decay=5e-5, alpha=kd_alpha, temperature=temp)
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
        mlflow.pytorch.log_model(teacher, "teacher")


if __name__ == "__main__":
    typer.run(main)
