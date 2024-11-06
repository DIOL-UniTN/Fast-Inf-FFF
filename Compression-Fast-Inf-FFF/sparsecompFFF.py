import torch
import numpy as np
import torch.nn as nn

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

def compute_sparsity_for_layers(layer_list, verbose=False):
    """Compute sparsity for each layer in a list of layers."""
    sparsity_info = []

    for layer in layer_list:
        weight = layer.data
        total_elements = weight.numel()
        zero_elements = (weight == 0).sum().item()
        sparsity = zero_elements / total_elements
        sparsity_info.append((layer.__class__.__name__, sparsity, total_elements, zero_elements))
    
    # Print the sparsity information for each layer
    if verbose:
        for layer, sparsity, total_elements, zero_elements in sparsity_info:
            print(f'Layer: {layer}, Sparsity: {1-sparsity:.4f}, Total Elements: {total_elements}, Zero Elements: {zero_elements}')

    return sparsity_info

def print_size_model(model, list_of_fc_layers, list_of_fc_sparsity, verbose=False):
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
    verbose and print("-------------------------------------------------------------------------------------------")
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
        weight = fc_layer.nelement() * fc_layer.element_size()
        
        # save in no sparsity
        total_size_no_sparsity += weight
        
        # set sparsity
        weight = min(1, 2 * sparsity) * weight
        
        # FROM Representation
        if (sparsity <= 0.5): # Dipende dall'analisi che vuoi fare
            if (len(list(fc_layer.shape)) == 3):
                verbose and print("Layer require additional", fc_layer.shape[0], "variables, total size with 4 bytes:", fc_layer.shape[0]*4 / kb)
                total_size_with_sparsity_CSC += (fc_layer.shape[0]*4) # number of filter
            elif (len(list(fc_layer.shape)) == 2):
                total_size_with_sparsity_CSC += (fc_layer.shape[1] + 1)*4 # number of column
                verbose and print("Layer require additional", fc_layer.shape[1]+1, "variables, total size with 4 bytes:", (fc_layer.shape[1]+1)*4 / kb)
            
        total_size_with_sparsity_CSC += weight
        
        # save in with sparsity
        total_size_with_sparsity += weight
        
        size_layer_list.append(weight)
        
        # print total - print weight - print bias
        verbose and print("Layer "+str(num)+":\t\t", (weight) / kb,
              "KB, \tweight:\t", weight / kb, "KB")
    
    # print total no sparisty
    verbose and print("Size FC Layer (no sparsity):\t", total_size_no_sparsity / kb,"KB")
    
    # print total with sparsity
    verbose and print("Size FC Layer (with sparsity):\t", total_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparsity
    verbose and print("Total Size no sparsity:\t\t", model_size_no_sparsity / kb ,"KB")
    
    # print model total - total no sparisty + total with sparsity
    model_size_with_sparsity = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity
    verbose and print("Total Size with sparsity:\t", model_size_with_sparsity / kb,"KB")
    
    # print model total - total no sparisty + total with sparsity and CSC
    model_size_with_sparsity_CSC = model_size_no_sparsity - total_size_no_sparsity + total_size_with_sparsity_CSC
    verbose and print("Total Size with sparsity and CSC representation:\t", model_size_with_sparsity_CSC / kb,"KB")
    
    verbose and print("-------------------------------------------------------------------------------------------")
    
    return model_size_with_sparsity, model_size_with_sparsity_CSC, size_layer_list

def perform_compression(model, list_of_fc_layers, list_of_fc_sparsity, learning_rate, num_epochs, train_loader,
                        test_loader,model_device,val_loader=None, model_name=None, given_criterion=None,
                        calculate_inputs=None,calculate_outputs=None, history=False, regularizerParam = 0,
                        fastInfLoss= False, fastInfNormWeight=0.0):
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
            #if (sparseTraining):
            print("Activating:", fc_layer.shape)
            fc_layer.requires_grad = True
        elif (sparsity > 0):
            print("Activating:", fc_layer.shape)
            fc_layer.requires_grad = True
        else:
            fc_layer.weight = torch.nn.Parameter(torch.zeros_like(fc_layer.weight), requires_grad=False)
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
                if fastInfLoss and fastInfNormWeight != 0:
                    l2loss = 0.0
                    if hasattr(model, 'fff'):
                        l2loss += model.fff.w1s.pow(2).sum()
                        l2loss += model.fff.w2s.pow(2).sum()
                    else:
                        for x in model.parameters():
                            l2loss += x.pow(2).sum()
                    loss += fastInfNormWeight * l2loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # apply hardthreshold - in the list we have only layer with require_grad = True
                for fc_layer, sparsity in zip(list_of_fc_layers, list_of_fc_sparsity):
                    layer = fc_layer.data
                    new_layer = hardThreshold(layer, sparsity)
                    with torch.no_grad():
                        fc_layer.data = torch.FloatTensor(new_layer).to(model_device)
                
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

def compress_FF_models(model, target_size, train_loader, test_loader, val_loader=None, num_epochs=10, learning_rate = 0.001, criterion=None, fastInfLoss=False, fastInfNormWeight=0.0, compressionStep = 0.1, fastCompression = True, modelName = "compressed_model", device="cpu"):
    # check model
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert target_size > 0, "The target size (kB) should be greater than 0"
    assert fastInfNormWeight >= 0, "Norm weight should be equal or greater than 0"
    assert compressionStep > 0, "Compression step should be equal or greater than 0"
    
    print("Target size requested:", target_size, "KB")
    
    model = model.to(device)
    
    # get sparsity of model
        # for layers with sparsity less than 1/2, consider as full
        
    model.train()
    layers_list = []
    for (name, p) in (model.named_parameters()):
        if (len(list(p.shape)) > 1 and p.requires_grad):
            layers_list.append(p)
    current_sparsity = compute_sparsity_for_layers(layers_list)
    sparsity_list = []
    for x, i in enumerate(current_sparsity):
        sp = 1-i[1]
        if sp >= 0.5:
            sp = 1
        sparsity_list.append(sp)
    print("Starting Density of model's parameters:", sparsity_list)
    
    # get sizes of model
    un, comp, layers = print_size_model(model, layers_list, sparsity_list)
    layers_sizes = layers.copy()
    
    starting_size = un if comp > un else comp

    print("Starting size of the model:", starting_size / 1000, "KB")
    
    if (target_size*1000 >= starting_size):
        print("Target size is already met! No compression performed")
        return starting_size
    
    # Compression step, as we get closer to target size we can reduce the size of the step
    # smaller step will require more iteration but will end with closer size to the target
    initial_step = 0.5
    step_decay = 0.1
    
    # Training Details
    if (criterion == None):
        criterion = nn.CrossEntropyLoss()
    
    # End compression
    final_size = comp
    end = False
    
    # start compressing
    for i in range(1, 100):
        # get index (depth) of largest layer
        index_of_largest = np.argmax(layers_sizes)
        current_sparsity = sparsity_list[index_of_largest]
        
        # reduce layers - step
        if (current_sparsity == 1):
            sparsity_list[index_of_largest] = initial_step - (initial_step * compressionStep)
        else:
            sparsity_list[index_of_largest] = current_sparsity - (current_sparsity * compressionStep)
        un, comp, layers = print_size_model(model, layers_list, sparsity_list)
        
        # we compress once we selected the target sparsity values if we want a faster compression time
        # otherwise we can compress at each step, lower epochs are suggested
        if (comp / 1000 < target_size or not fastCompression):
            # compress and save result
            MODEL_NAME_COMPRESSED = modelName + "_" + str(round(comp / 1000))
            model.train()
            accuracy = perform_compression(model, layers_list, sparsity_list, learning_rate, num_epochs,
                                           train_loader, test_loader, device,
                                           val_loader=val_loader, model_name=MODEL_NAME_COMPRESSED, given_criterion=criterion,
                                           fastInfLoss=fastInfLoss, fastInfNormWeight=fastInfNormWeight)
            
            # Load best model saved during compression
            model.load_state_dict(torch.load(MODEL_NAME_COMPRESSED+".h5", map_location='cpu'))
            model.to(device)
            model.eval()
            
        if (comp / 1000 < target_size):
            final_size = comp
            end = True
            break
        
        # continue reducing sparsity
        print(i, "iteration - ", "Size:", comp, sparsity_list)
    
    return final_size