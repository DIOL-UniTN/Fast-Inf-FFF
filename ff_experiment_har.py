import typer
import pickle
import mlflow
import numpy as np
from tqdm import trange
from fff_trainer import FF, train, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class HarDataset(Dataset):
    def __init__(self, fold="train"):
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


def main(layer_width: int, epochs: int):
    trainloader, testloader, _ = load_data()
    net = FF(300, layer_width, 10).to(DEVICE)

    with mlflow.start_run(experiment_id="20"):
        mlflow.log_param("leaf_width", layer_width)
        mlflow.log_param("depth", 1)
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
        mlflow.pytorch.log_model(net, "model")


if __name__ == "__main__":
    typer.run(main)
