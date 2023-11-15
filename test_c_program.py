import sys
from fff_experiment_mnist import load_data
trainloader, testloader, n = load_data()

data = None
labels = None
for x, y in testloader:
    data = x
    labels = y
data = data.cpu().detach().numpy()
labels = labels.cpu().detach().numpy()

data = data.reshape(len(data), -1)

sys.stdout = open("data.h", "w")
print("""
#ifndef DATA_H_
#define DATA_H_
#include "fixed.h"
#include "mem.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

"""
)
print("__hifram fixed INPUTS["+str(len(x))+"][INPUT_SIZE] = {")
print(", ".join([str(val) for val in x.flatten().cpu().numpy()]))
print("}")

print("__hifram fixed OUTPUTS["+str(len(x))+"] = {")
print(", ".join(str(val) for val in y.flatten().cpu().numpy()))
print("}")
# for i, (x, y) in enumerate(zip(data, labels)):
#     # print(f"./a.out {' '.join(map(str, x))} && echo {y}")

print("#endif DATA_H_")
