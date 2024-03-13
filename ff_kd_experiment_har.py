import torch
import typer
import pickle
import mlflow
import numpy as np
from tqdm import trange
from fff_trainer import FF, train_kd, test, DEVICE
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

TEACHER_RUNIDS = {512: "8b0567b0f3914f45be24108ce6ea7bcc", # layer_width: run_id
                  256: "568ddc007d95461896f8976c279f7434",
                  128: "a2743fd29dee4ba3b9c5711306b96aad",
                  64: "55bcd9b62dde43139fa9a37ca1aa4a7e",
                  }
def main(layer_width: int, epochs: int, teacher_layer_width:int, kd_alpha:float, temp:float):
    trainloader, testloader, _ = load_data()
    net = FF(300, layer_width, 6).to(DEVICE)
    mlflow.artifacts.download_artifacts(run_id=TEACHER_RUNIDS[teacher_layer_width], 
                                        artifact_path='model/data/model.pth', dst_path=".")
    teacher = torch.load('model/data/model.pth').to(DEVICE)
    with mlflow.start_run(experiment_id="27"):
        mlflow.log_param("leaf_width", layer_width)
        mlflow.log_param("depth", 1)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("teacher_layer_width", teacher_layer_width)
        mlflow.log_param("teacher_run_id", TEACHER_RUNIDS[teacher_layer_width])
        mlflow.log_param("kd_alpha", kd_alpha)
        mlflow.log_param("temperature", temp)

        # Train the net and log on mlflow
        for i in trange(epochs):
            train_kd(net, teacher, trainloader, 1, 0.0, alpha=kd_alpha, temperature=temp)
            train_loss, train_acc = test(net, trainloader)
            test_loss, test_acc = test(net, testloader)

            mlflow.log_metric("train_accuracy", train_acc, step=i)
            mlflow.log_metric("train_loss", train_loss, step=i)
            mlflow.log_metric("test_accuracy", test_acc, step=i)
            mlflow.log_metric("test_loss", test_loss, step=i)

        # Log model
        mlflow.pytorch.log_model(net, "model")
        mlflow.pytorch.log_model(teacher, "teacher")

if __name__ == "__main__":
    typer.run(main)
