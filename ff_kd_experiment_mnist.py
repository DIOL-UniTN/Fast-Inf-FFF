import typer
import mlflow
from tqdm import trange
from fff_trainer import FF, train_kd, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

def load_data():
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trainset = MNIST("./data", train=True,  download=True, transform=transform)
    testset = MNIST("./data",  train=False, download=True, transform=transform)

    # Select class to keep 
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    testloader = DataLoader(testset, batch_size=1024)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

TEACHER_RUNIDS = {64: "afdf22660d7f4ce7b62ac6ba783e20da", # layer_width: run_id
                  32: "b13f02f0cbdc424eb006f8fc6e4502a7",
                  16: "e9be2bc6dc424610a12bf673d41c938c",
                  }
def main(layer_width: int, epochs: int, teacher_layer_width:int, kd_alpha:float, temp:float):
    trainloader, testloader, _ = load_data()
    net = FF(784, layer_width, 10).to(DEVICE)
    mlflow.artifacts.download_artifacts(run_id=TEACHER_RUNIDS[teacher_layer_width], 
                                        artifact_path='model/data/model.pth', dst_path=".")
    teacher = torch.load('model/data/model.pth').to(DEVICE)
    with mlflow.start_run(experiment_id="19"):
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
