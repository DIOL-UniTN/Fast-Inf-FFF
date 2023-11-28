import typer
import mlflow
from tqdm import trange
from fff_trainer import Net, train, test, DEVICE, ViTFFF
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def load_data():
    """Load CIFAR (training and test set)."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Select class to keep 
    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
    testloader = DataLoader(testset, batch_size=1024)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples


def main(leaf_width: int, depth: int, epochs: int, norm_weight: float, latent_size: int):
    trainloader, testloader, _ = load_data()
    net = ViTFFF((3, 4, 4), leaf_width, latent_size, 10, depth).to(DEVICE)

    with mlflow.start_run(experiment_id="8"):
        mlflow.log_param("leaf_width", leaf_width)
        mlflow.log_param("depth", depth)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("norm_weight", norm_weight)
        mlflow.log_param("hardened", False)
        mlflow.log_param("latent_size", latent_size)

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
