import typer
import mlflow
from tqdm import trange
from fff_trainer import BFF, train, test, DEVICE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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


def main(layer_width: int, epochs: int, quant_in: bool, quant_out: bool):
    trainloader, testloader, _ = load_data()
    net = BFF(784, layer_width, 10, quant_in, quant_out).to(DEVICE)
    mlflow.log_param("is out layer binarized", quant_out)
    mlflow.log_param("is input binarized", quant_in)

    with mlflow.start_run(experiment_id="29"):
        mlflow.log_param("hidden_width", layer_width)
        mlflow.log_param("is input binarized", quant_in)
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