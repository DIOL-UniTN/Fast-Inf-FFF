from collections import OrderedDict

import torch
import typer
from fastfeedforward import FFF
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import flwr as fl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
PERCENTAGE = 0.1

def load_data(client_id):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ]
    )
    trainset = MNIST("./data", train=True, download=True, transform=transform)
    testset = MNIST("./data", train=False, download=False, transform=transform)

    # Select class to keep 
    idx = trainset.targets == client_id
    for i in range(int(len(trainset.data) * PERCENTAGE)):
        idx[i] = not idx[i]
    trainset.targets = trainset.targets[idx]
    trainset.data = trainset.data[idx]
    idx = testset.targets == client_id
    testset.targets = testset.targets[idx]
    testset.data = testset.data[idx]

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128)

    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class Net(torch.nn.Module):
    def __init__(self, input_width, leaf_width, output_width, depth, activation, dropout, region_leak):
        super(Net, self).__init__()
        self.fff = FFF(input_width, leaf_width, output_width, depth, activation, dropout, region_leak)

    def forward(self, x):
        x = x.view(len(x), -1)
        return self.fff(x)

    def parameters(self):
        return self.fff.parameters()


class Client(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, num_examples):
        self._net = net
        self._trainloader = trainloader
        self._testloader = testloader
        self._num_examples = num_examples

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self._net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self._net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self._net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self._net, self._trainloader, epochs=1)
        loss, accuracy = test(self._net, self._trainloader)
        ret_dict = {"accuracy": accuracy, 'loss': loss}
        print(ret_dict)
        return self.get_parameters(config={}), self._num_examples["trainset"], ret_dict

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self._net, self._testloader)
        return float(loss), self._num_examples["testset"], {"accuracy": float(accuracy), "loss": float(loss)}


def main(client_id: int, l: int, d: int, serv_addr="0.0.0.0:8080"):
    net = Net(input_width=784, leaf_width=l, output_width=10, depth=d, activation=torch.nn.ReLU(), dropout=0.0, region_leak=0.0).to(DEVICE)
    trainloader, testloader, num_examples = load_data(client_id)

    fl.client.start_numpy_client(server_address=serv_addr, client=Client(net, trainloader, testloader, num_examples))


if __name__ == "__main__":
    typer.run(main)
