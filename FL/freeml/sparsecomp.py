import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def print_size_fc(model, list_of_fc_layers, list_of_fc_sparsity, verbose=False):
    '''
    model has to be sublass of nn.Module
        check the subclass with: issubclass(sub, sup), return true if sub is sublcass of sup
                                 isinstance(sub_instance, sup), return true if is sub_instance is subclass of sup
    list_of_fc_layers: list of fully connected layer OF THE MODEL (should be a pointer to layer of model)
    list_of_fc_sparsity: list of the sparsity for each fully connected layer
    '''
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    kb = 1000
    print("-------------------------------------------------------------------------------------------")
    model_size_no_sparsity = 0
    for param in model.parameters():
        model_size_no_sparsity += param.nelement() * param.element_size()
    for buffer in model.buffers():
        model_size_no_sparsity += buffer.nelement() * buffer.element_size()
    
    total_size_no_sparsity = 0
    total_size_with_sparsity = 0
    total_size_with_sparsity_CSC = 0
    
    size_layer_list = []
    num = 0
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        num += 1
        # get size
        verbose and print("Layer " + str(num), fc_layer)
        weight = fc_layer.weight.nelement() * fc_layer.weight.element_size()
        bias = 0
        if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
            bias = fc_layer.bias.nelement() * fc_layer.bias.element_size()
        
        # save in no sparsity
        total_size_no_sparsity += weight + bias
        
        # set sparsity
        weight = min(1, 2 * sparsity) * weight
        
        # FROM Representation
        if (sparsity <= 0.5): # Dipende dall'analisi che vuoi fare
            if (isinstance(fc_layer, torch.nn.Conv2d)):
                verbose and print("Layer require additional", fc_layer.weight.shape[0], "variables, total size with 4 bytes:", fc_layer.weight.shape[0]*4 / kb)
                total_size_with_sparsity_CSC += (fc_layer.weight.shape[0]*4) # number of filter
            elif (isinstance(fc_layer, torch.nn.Linear)):
                total_size_with_sparsity_CSC += (fc_layer.weight.shape[1] + 1)*4 # number of column
                verbose and print("Layer require additional", fc_layer.weight.shape[1]+1, "variables, total size with 4 bytes:", (fc_layer.weight.shape[1]+1)*4 / kb)
            
        total_size_with_sparsity_CSC += weight + bias
        
        # save in with sparsity
        total_size_with_sparsity += weight + bias
        
        size_layer_list.append(weight + bias)
        
        # print total - print weight - print bias
        print("Layer "+str(num)+":\t\t", (weight + bias) / kb,
              "KB, \tweight:\t", weight / kb,
              "KB, \tbias:", bias / kb, "KB")
    
    # print total no sparisty
    print("Size FC Layer (no sparsity):\t", total_size_no_sparsity / kb,"KB")
    
    # print total with sparsity
    print("Size FC Layer (with sparsity):\t", total_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparsity
    print("Total Size no sparsity:\t\t", model_size_no_sparsity / kb ,"KB")
    
    # print model total - total no sparisty + total with sparsity
    model_size_with_sparsity = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity
    print("Total Size with sparsity:\t", model_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparisty + total with sparsity and CSC
    model_size_with_sparsity_CSC = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity_CSC
    print("Total Size with sparsity and CSC representation:\t", model_size_with_sparsity_CSC / kb,"KB")
    
    print("-------------------------------------------------------------------------------------------")
    
    return model_size_with_sparsity, model_size_with_sparsity_CSC, size_layer_list

def print_full_model(model):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    kb = 1000
    model_size = 0
    for name, param in model.named_parameters():
        layer_size = param.nelement() * param.element_size()
        model_size += layer_size
        print(name,"\t", param.nelement(), "\t", param.element_size(),"\t", layer_size / kb, "KB")
        
    for name, buffer in model.named_buffers():
        layer_size = buffer.nelement() * buffer.element_size()
        model_size += layer_size
        print(name,"\t", layer_size / kb, "KB")
    print("Model Size:", model_size / kb, "KB")

def hardThreshold(A: torch.Tensor, sparsity):
    '''
    Given a Tensor A and the correponding sparsity, returns a copy in the
    format of numpy array with the constraint applied
    '''
    matrix_A = A.data.cpu().detach().numpy().ravel()    
    if len(matrix_A) > 0:
        threshold = np.percentile(np.abs(matrix_A), (1 - sparsity) * 100.0, method='higher')
        matrix_A[np.abs(matrix_A) < threshold] = 0.0
    matrix_A = matrix_A.reshape(A.shape)
    return matrix_A
    
def perform_compression(model, list_of_fc_layers, list_of_fc_sparsity, learning_rate, num_epochs, train_loader,
                        test_loader,model_device,val_loader=None, model_name=None, given_criterion=None,
                        calculate_inputs=None,calculate_outputs=None, history=False, regularizerParam = 0):
    '''
    model has to be sublass of nn.Module
        check the subclass with: issubclass(sub, sup), return true if sub is sublcass of sup
                                 isinstance(sub_instance, sup), return true if is sub_instance is subclass of sup
    list_of_fc_layers: list of fully connected layer OF THE MODEL (should be a pointer to layer of model)
    list_of_fc_sparsity: list of the sparsity for each fully connected layer
    NOTE - Sparsity applied only to weight of FC, not on bias
    NOTE - The list are modified during execution, so are copied with list.copy() to avoid changing the original list
    '''
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    # asset sparsity between 0 and 1
    valid_sparsity = True
    for sparsity in list_of_fc_sparsity:
        if (sparsity > 1) or (sparsity < 0):
            valid_sparsity = False
    assert valid_sparsity, "The sparsity value must be between 0 and 1"
    list_of_fc_layers = list_of_fc_layers.copy()
    list_of_fc_sparsity = list_of_fc_sparsity.copy()
    # The idea is get the model, set all parameter to not require gradient, set fully connected layer to require gradient,
    # perform training
    
    # disabling parameters
    for name, param in model.named_parameters():
        print("Disabling:", name)
        param.requires_grad = False
    
    # activating fully connected layers only if its sparsity is > 0
    # if a layer has sparsity equal to zero we can override with 0
    # if all sparsity is set to 1, compression is not requested
    sparseTraining = False
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        if (sparsity == 1):
            if (sparseTraining):
                print("Activating:", fc_layer)
                fc_layer.weight.requires_grad = True
                if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                    fc_layer.bias.requires_grad = True
        elif (sparsity > 0):
            print("Activating:", fc_layer)
            fc_layer.weight.requires_grad = True
            if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                fc_layer.bias.requires_grad = True
        else:
            fc_layer.weight = torch.nn.Parameter(torch.zeros_like(fc_layer.weight), requires_grad=False)
            if (hasattr(fc_layer, 'bias') and fc_layer.bias is not None):
                fc_layer.bias.requires_grad = True
            # delete from the list (since no need to update them)
            list_of_fc_layers.remove(fc_layer)
            list_of_fc_sparsity.remove(sparsity)
            
        if (sparsity < 1):
            sparseTraining = True
    
    acc = 0
    # TEST - compute accuracy
    accuracyHistory = []
    lastCorrect = 0
    totalPredictions = 0
    numberOfUpdates = len(test_loader)
        
    if not (sparseTraining):
        print("No need to perform compression, all layers's sparsity is set to 1")
    else: # PERFORM TRAINING - COMPRESSION
        
        # set up
        criterion = nn.NLLLoss()
        if given_criterion:
            criterion = given_criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        n_total_steps = len(train_loader)
        
        # to save best results
        best_val_epoch, best_val_loss, best_val_acc, best_acc_epoch = 0, 1e6, 0, 0
        
        for epoch in range(num_epochs):
            
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                # origin shape: [100, 1, 28, 28]
                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                                
                # Forward pass
                
                # preforward
                if calculate_inputs:
                    inputs = calculate_inputs(inputs)
                
                # forward
                if calculate_outputs:
                    outputs = calculate_outputs(inputs)
                else:
                    outputs = model.forward(inputs)
                
                # Regularization
                regularizer = 0
                if (regularizerParam != 0):
                    for layer in list_of_fc_layers:
                        regularizer += (torch.norm(layer.weight)**2)
                # Loss
                loss = criterion(outputs, labels) + (regularizer * regularizerParam)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # apply hardthreshold - in the list we have only layer with require_grad = True
                for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
                    layer = fc_layer.weight.data
                    new_layer = hardThreshold(layer, sparsity)
                    with torch.no_grad():
                        fc_layer.weight.data = torch.FloatTensor(new_layer).to(model_device)
                
                # print Accuracy
                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
            # Use Validation Set at each epochs to pick the most 
            if (val_loader and model_name):
                model.eval()
                with torch.no_grad():
                    v_loss = 0
                    n_correct = 0
                    n_samples = 0
                    n_iterations = 0
                    for inputs, labels in test_loader:
                        inputs = inputs.to(model_device)
                        labels = labels.to(model_device)
                        # Forward pass
                
                        # preforward
                        if calculate_inputs:
                            inputs = calculate_inputs(inputs)
                        outputs = 0 
                        # forward
                        if calculate_outputs:
                            outputs = calculate_outputs(inputs)
                        else:
                            outputs = model.forward(inputs)
                        
                        # for calculating v_loss
                        loss = criterion(outputs, labels)                       
                        v_loss += loss.item()
                        n_iterations += 1
                        
                        # max returns (value, index)
                        _, predicted = torch.max(outputs.data, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()
                    
                    # Val test completed, now checking the results
                    v_loss = v_loss/(n_iterations)
                    v_loss = round(v_loss, 5)
                    v_acc = round(100*(n_correct / n_samples), 5)
                    
                    if v_acc >= best_val_acc:
                        torch.save(model.state_dict(), model_name+"_acc.h5")
                        best_acc_epoch = epoch + 1
                        best_val_acc = v_acc
                    if v_loss <= best_val_loss:
                        torch.save(model.state_dict(), model_name+".h5")
                        best_val_epoch = epoch + 1
                        best_val_loss = v_loss
                    #print(f'Epoch[{epoch+1}]: t_loss: {t_loss} t_acc: {t_acc} v_loss: {v_loss} v_acc: {v_acc}')
                    print(f'Epoch[{epoch+1}]: v_loss: {v_loss} v_acc: {v_acc}')
        
        
        # Use Validation Set at each epochs to pick the most 
        if (val_loader and model_name):
            model.load_state_dict(torch.load(model_name+".h5", map_location='cpu'))
            print('Best model saved at epoch: ', best_val_epoch)
            print('Best acc model saved at epoch: ', best_acc_epoch)
        
        # USING TEST SET TO CHECK ACCURACY
        #model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(model_device)
                labels = labels.to(model_device)
                   # Forward pass
                
                # preforward
                if calculate_inputs:
                    inputs = calculate_inputs(inputs)
                outputs = 0 
                # forward
                if calculate_outputs:
                    outputs = calculate_outputs(inputs)
                else:
                    outputs = model.forward(inputs)
                # max returns (value, index)
                
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()                
            acc = 100.0 * n_correct / n_samples
            totalPredictions = n_samples
            print(f'Accuracy of the network on the 10000 test images: {acc} %')

        
    result = {
        'correctPredictions': lastCorrect,
        'totalPredictions': totalPredictions,
        'accuracyThroughEpochs': accuracyHistory,
        'numberOfUpdate': numberOfUpdates,
    }
    
    return acc

def get_layers(model):
    """Recursively get all layers in a PyTorch model."""
    list_layers = []
    # for name, module in model.named_children():
    #     # check type of module
    #     is_conv1d = isinstance(module, torch.nn.Conv1d)
    #     is_conv2d = isinstance(module, torch.nn.Conv2d)
    #     is_linear = isinstance(module, torch.nn.Linear)
    #     is_sequential = isinstance(module, torch.nn.Sequential)
    #     if (is_conv1d or is_conv2d or is_linear):
    #         list_layers.append(module)
    #     if (is_sequential):
    #         for sub_name, sub_module in module.named_children():
    #             print(sub_name)
    #             # check type of module
    #             is_conv1d = isinstance(sub_module, torch.nn.Conv1d)
    #             is_conv2d = isinstance(sub_module, torch.nn.Conv2d)
    #             is_linear = isinstance(sub_module, torch.nn.Linear)
    #             if (is_conv1d or is_conv2d or is_linear):
    #                 list_layers.append(sub_module)
    for layer in model.children():
        if isinstance(layer, nn.Sequential):
            # If it's a sequential container, recursively get its layers
            list_layers.extend(get_layers(layer))
        else:
            # If it's a single layer, add it to the list
            if (isinstance(layer, torch.nn.Conv1d) or isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear)):
                list_layers.append(layer)
    return list_layers

def apply_sparsity(model, list_of_fc_layers, list_of_fc_sparsity, model_device):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert len(list_of_fc_layers) == len(list_of_fc_sparsity), "The lists should be of the same length"
    # asset sparsity between 0 and 1
    valid_sparsity = True
    for sparsity in list_of_fc_sparsity:
        if (sparsity > 1) or (sparsity < 0):
            valid_sparsity = False
    assert valid_sparsity, "The sparsity value must be between 0 and 1"
    
    list_of_fc_layers = list_of_fc_layers.copy()
    list_of_fc_sparsity = list_of_fc_sparsity.copy()
    
    # apply hardthreshold - in the list we have only layer with require_grad = True
    for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
        layer = fc_layer.weight.data
        new_layer = hardThreshold(layer, sparsity)
        with torch.no_grad():
            fc_layer.weight.data = torch.FloatTensor(new_layer).to(model_device)
    
def calculate_accuracy(model, train_loader, test_loader, model_device, calculate_inputs=None, calculate_outputs=None):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"

    acc = 0

    # TEST - compute accuracy
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            # preforward
            if calculate_inputs:
                inputs = calculate_inputs(inputs)
            outputs = 0 
            # forward
            if calculate_outputs:
                outputs = calculate_outputs(inputs)
            else:
                outputs = model.forward(inputs)
                    
            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the train images: {acc} %')

        n_correct = 0
        n_samples = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(model_device)
            labels = labels.to(model_device)
            # preforward
            if calculate_inputs:
                inputs = calculate_inputs(inputs)
            outputs = 0 
            # forward
            if calculate_outputs:
                outputs = calculate_outputs(inputs)
            else:
                outputs = model.forward(inputs)
            
            # max returns (value, index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')

    return acc

def compute_sparsity_for_layers(layer_list):
    """Compute sparsity for each layer in a list of layers."""
    sparsity_info = []

    for layer in layer_list:
        if hasattr(layer, 'weight'):
            weight = layer.weight.data
            total_elements = weight.numel()
            zero_elements = (weight == 0).sum().item()
            sparsity = zero_elements / total_elements
            sparsity_info.append((layer.__class__.__name__, sparsity, total_elements, zero_elements))
    
    # Print the sparsity information for each layer
    for layer, sparsity, total_elements, zero_elements in sparsity_info:
        print(f'Layer: {layer}, Sparsity: {1-sparsity:.4f}, Total Elements: {total_elements}, Zero Elements: {zero_elements}')

    return sparsity_info