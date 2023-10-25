import torch
import typer
from fastfeedforward import FFF


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")


def train(net, trainloader, epochs, norm_weight=0.0):
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
            if norm_weight != 0:
                loss += norm_weight * net.fff.w1s.pow(2).sum()
                loss += norm_weight * net.fff.w2s.pow(2).sum()
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
        self.fff = FFF(input_width, leaf_width, output_width, depth, torch.nn.ReLU(), dropout, train_hardened=True, region_leak=region_leak)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.fff(x)
        x = torch.nn.functional.softmax(x, -1)
        return x

    def parameters(self):
        return self.fff.parameters()


class FF(torch.nn.Module):
    def __init__(self, input_width, layer_width, output_width):
        super(FF, self).__init__()
        self.fc1 = torch.nn.Linear(input_width, layer_width)
        self.fc2 = torch.nn.Linear(layer_width, output_width)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.softmax(self.fc2(x), -1)
        return x

    def parameters(self):
        return [*self.fc1.parameters(), *self.fc2.parameters()]


def compute_n_params(input_width: int, l_w: int, depth: int, output_width: int):
    fff = Net(input_width, l_w, output_width, depth, 0, 0)
    ff = FF(input_width, l_w, output_width)

    n_ff = 0
    n_fff = 0
    for p in ff.parameters():
        n_ff += p.numel()
    for i, p in enumerate(fff.parameters()):
        print(f"[{i}-th layer]: {p.shape}")
        n_fff += p.numel()

    print(f"FFF: {n_fff}\nFF: {n_ff}")


def main():
    typer.run(compute_n_params)


if __name__ == "__main__":
    main()
