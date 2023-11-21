import sys
from fff_experiment_mnist import load_data
trainloader, testloader, n = load_data()

data = None
labels = None
for x, y in testloader:
    data = x
    labels = y
    break
data = data.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

data = data.reshape(len(data), -1)

for i in range(len(data) // 10):
    sys.stdout = open(f"data_batch_{i}.h", "w")
    print("""
#ifndef DATA_H_
#define DATA_H_
#include "fixed.h"
#include "mem.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

    """
    )
    print("__hifram fixed INPUTS[10][INPUT_SIZE] = {")
    print(", ".join([str(val) for val in data[i*10:(i+1)*10].flatten()]))
    print("}")

    print("__hifram fixed OUTPUTS[10] = {")
    print(", ".join(str(val) for val in labels[i*10:(i+1)*10].flatten()))
    print("}")
    # for i, (x, y) in enumerate(zip(data, labels)):
    #     # print(f"./a.out {' '.join(map(str, x))} && echo {y}")

    print("#endif DATA_H_")
