import typer
import mlflow
import numpy as np
from tqdm import trange
from fff_trainer import BFF, train, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from dagshub.data_engine import datasources
import pandas as pd
import pickle

class HarDataset(Dataset):
    def __init__(self, fold="train"):
        self.data_v2 = pd.read_csv("data/har_v2/train.csv")
        self.data = pickle.load(open(f"data/har/{fold}_data.summary", "rb"), encoding="latin1")
        self.labels = pickle.load(open(f"data/har/{fold}_labels.summary", "rb"), encoding="latin1")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        y = np.argmax(y, axis=0)
        return x, y


def load_data():
    trainset = HarDataset("train")
    testset = HarDataset("test")

    # Select class to keep 
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    testloader = DataLoader(testset, batch_size=1024)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def main(layer_width: int, epochs: int, quant_in: bool, quant_out: bool):
    ds = datasources.get('leocus4/TinyFFF', 'HAR')
    ds.all().download_files(target_dir='')
    trainloader, testloader, _ = load_data()
    net = BFF(300, layer_width, 10, quant_in, quant_out).to(DEVICE)

    with mlflow.start_run(experiment_id="30"):
        mlflow.log_param("hidden_width", layer_width)
        mlflow.log_param("is input binarized", quant_in)
        mlflow.log_param("is out layer binarized", quant_out)
        mlflow.log_param("epochs", epochs)

        # Train the net and log on mlflow
        for i in trange(epochs):
            train(net, trainloader, 1)
            train_loss, train_acc = test(net, trainloader)
            test_loss, test_acc = test(net, testloader)

            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)

        # Log model
        mlflow.pytorch.log_model(net, "bmodel")


if __name__ == "__main__":
    typer.run(main)
