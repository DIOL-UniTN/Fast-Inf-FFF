import typer
import torch
import numpy as np
from tqdm import trange, tqdm
from fff_trainer import FF, train, test, DEVICE
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_size(params):
    n = 0
    for p in params.values():
        n += p.count_nonzero()
    return n * 2


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


def main(layer_width: int, epochs: int):
    trainloader, testloader, _ = load_data()
    net = FF(32*32*3, layer_width, 10).to(DEVICE)

    for i in trange(epochs):
        train(net, trainloader, 1)
        # train_loss, train_acc = test(net, trainloader)
        # test_loss, test_acc = test(net, testloader)

        # print("train_accuracy", train_acc)
        # print("train_loss", train_loss)
        # print("test_accuracy", test_acc)
        # print("test_loss", test_loss)

    test_loss, test_acc = test(net, testloader)
    print("test_accuracy before pruning", test_acc)

    params = net.state_dict()
    n_params = 0
    for p in params.values():
        n_params += p.numel()
    print(n_params, "parameters")
    for i in range(1):
        k = (n_params - 60*1024)/n_params
        # train(net, trainloader, 1)
        params = net.state_dict()
        zeroed = torch.Tensor(0)
        # for i in range(n_params // 10, n_params - 40 * 1024, n_params // 10):
        for i in range(int(n_params * k)):
            min = float("inf")
            argmin = None

            for i, p in params.items():
                tmpmin = p[torch.where(p != 0)].abs().min()
                if tmpmin < min:
                    min = tmpmin
                    argmin = i, torch.where(p.abs() == min)

            zeroed = params[argmin[0]][argmin[1]] * 0
            params[argmin[0]][argmin[1]] = zeroed
        net.load_state_dict(params)
        # for n, p in params.items():
        #     try:
        #         p[torch.where(p.abs() < 1e-3)] = zeroed
        #     except Exception as e:
        #         print(e)
        #         pass
        if (get_size(params) <= 60 * 1024):
            break

    test_loss, test_acc = test(net, testloader)
    print("test_accuracy after pruning", test_acc)
    n_params = 0
    for p in params.values():
        n_params += p.count_nonzero()
    print(n_params, "parameters left")


if __name__ == "__main__":
    typer.run(main)


"""
Run 1:
test_accuracy before pruning 0.4246
test_accuracy after pruning 0.41
Run 2:
test_accuracy before pruning 0.4237
test_accuracy after pruning 0.4194
Run 3:
test_accuracy before pruning 0.4192
test_accuracy after pruning 0.4146
"""
