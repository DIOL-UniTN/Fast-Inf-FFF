import torch
import typer
from fastfeedforward import FFF


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # Train the network for the given number of epochs
    for _ in range(epochs):
        # Iterate over data
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    # Train the network for the given number of epochs
    with torch.no_grad():
        # Iterate over data
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
    def __init__(self, input_width, leaf_width, output_width, depth, dropout, region_leak):
        super(Net, self).__init__()
        self.fff = FFF(input_width, leaf_width, output_width, depth, torch.nn.ReLU(), dropout, region_leak)

    def forward(self, x):
        x = x.view(len(x), -1)
        return self.fff(x)

    def parameters(self):
        return self.fff.parameters()
