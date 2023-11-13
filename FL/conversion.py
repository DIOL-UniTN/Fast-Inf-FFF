import scipy.sparse as sp
import torch
import torch.nn as nn
import re
import os

'''
    The conversion to c is performed via these steps:
        1) Unrolling the model: all the Sequential layers are analyzed recursively,
           creating an ordered list of all the different layers of the model. This
           list is used as a forward function by applying each layer to the input
           which will gradually change shape arriving at the final output.
           This method was used to allow each layer to have the details of the output
           dimensions, necessary for execution in C.
           This method however has problems with models whose forward function does
           not correspond to the list of layers e.g. early exit or trees, so modifications
           may be necessary.
        2) For each element of the list:
            a) the type of layer is analysed,
            b) applied to the output of the previous layer,
            c) details of the layer are concatenated to the different strings: declaration
                and definition of parameters (model.c),
            d) the layers that contain stored parameters are printed following the correct
                representation in the corresponding files (e.g. fc1.h),
        3) Once all the layers have been analyzed, the model.c file is written using
            the stored strings (2.c), the input used is saved in input.h, and all the files
            that require to be imported (e.g. fc1.h) are written in the model.h file.

    Current Problem --- The save_compressed_model references variable used in prepare_input
    It happened since it was written originally in notebook and due to time constraints it hasn't
    be fixed yet. See the usage example to see how to get around the problem
'''

'''
    to be used before the model conversion
'''
def prepare_input(examples):
    number_of_examples = 1
    example_data, example_targets = next(examples)
    example_data_tensor = example_data
    example_data = example_data.numpy()
    example_targets = example_targets.numpy()
    searching_values = 9
    while (example_targets[0] != searching_values):
        example_data, example_targets = next(examples)
        example_data_tensor = example_data
        example_data = example_data.numpy()
        example_targets = example_targets.numpy()
    print("Example data:", example_data.shape)
    print("Example labels:", example_targets.shape)

    return example_data_tensor, example_data, example_targets

def pooling_declaration(module, param_name, num_this_layer_type, layer_output):
    # param_name = "param_maxPool_" + str(num_this_layer_type)
    
    intro = "/* Pooling layer " + str(num_this_layer_type) + " parameters */\n"
    
    outputDimension = -1
    outputShape = "{-1}"
    
    if (layer_output != None):
        outputDimension = len(layer_output.shape)
        print("Output Dimension:", outputDimension)
        outputShape = "{"
        for x in range(1, outputDimension):
            outputShape += str(layer_output.shape[x])
            if (x < outputDimension - 1):
                outputShape += ", "
            else:
                outputShape += "}"
    
    # Stride is INT or a touple of 2 int - HARDCODED to obtain 2 int
    stride = module.stride
    stride_list = []
    if isinstance(stride, int):
        stride_list.append(stride)
        stride_list.append(stride)
    else:
        for x in range(0, 2):
            stride_list.append(stride[x])
    strideString = "{" + str(stride_list[0]) +", " + str(stride_list[1]) + "}"
    
    # Kernel Size is INT or a touple of 2 int - HARDCODED to obtain 2 int
    k_size = module.kernel_size
    k_size_list = []
    if isinstance(k_size, int):
        k_size_list.append(k_size)
        k_size_list.append(k_size)
    else:
        for x in range(0, 2):
            k_size_list.append(k_size[x])
    sizeString = "{" + str(k_size_list[0]) +", " + str(k_size_list[1]) + "}"
    
    # ---------- layer parameters ---------- #
    pooling_t_declaration = "__fram pooling_t " + param_name + " = {\n"
    pooling_t_declaration += "\t.type = " + 'MAX' + ",\n"
    pooling_t_declaration += "\t.size = " + sizeString + ",\n"
    pooling_t_declaration += "\t.stride = " + strideString + ",\n"
    pooling_t_declaration += "\t.outputShape = " + outputShape + ",\n"
    pooling_t_declaration += "\t.outputDimension = " + str(outputDimension - 1) + ",\n"
    pooling_t_declaration += "};\n\n"
    
    return intro + pooling_t_declaration

def softmax_declaration(module, param_name, num_this_layer_type, layer_input):
    #param_name = "param_softmax_" + str(num_this_layer_type)
    
    intro = "/* Softmax layer " + str(num_this_layer_type) + " parameters */\n"
    
    input_num = -1
    
    if (layer_input != None):
        input_num = layer_input.shape[1]
    
    # ---------- layer parameters ---------- #
    softmax_t_declaration = "__fram softmax_t " + param_name + " = {\n"
    softmax_t_declaration += "\t.input_num = " + str(input_num) + ",\n"
    softmax_t_declaration += "};\n\n"
    
    return intro + softmax_t_declaration

def flatten_declaration(module, param_name, num_this_layer_type, layer_output):
    #param_name = "param_flatten_" + str(num_this_layer_type)
    
    intro = "/* Flatten layer " + str(num_this_layer_type) + " parameters */\n"
    
    new_shape_dim = -1
    new_shape_string = "{-1, -1}"
    
    if (layer_output != None):
        new_shape_dim = len(layer_output.shape)
        
        # HARDCODED: reversed order. From [1, 5, 1600] to [1600, 5, 1]
        new_shape_string = "{"
        for x in range(new_shape_dim - 1, 0 - 1, -1):
            new_shape_string += str(layer_output.shape[x])
            if (x > 0):
                new_shape_string += ", "
            else:
                new_shape_string += "}"
    
    # ---------- layer parameters ---------- #
    reshape_t_declaration = "__fram reshape_t " + param_name + " = {\n"
    reshape_t_declaration += "\t.new_shape = " + new_shape_string + ",\n"
    reshape_t_declaration += "\t.new_shape_dim = " + str(new_shape_dim) + ",\n"
    reshape_t_declaration += "};\n\n"
    
    return intro + reshape_t_declaration


def linear_declaration(module, param_name, num_this_layer_type, layer_output, next_activation, representation, layer_name, f, flit):
    
    weight_name = "weights_FC_" + str(num_this_layer_type)
    bias_name = "bias_FC_" + str(num_this_layer_type)
    #param_name = "param_fully_con_" + str(num_this_layer_type)
    
    # Check if there is a bias
    it_has_bias = False
    if (hasattr(module, 'bias') and module.bias is not None):
        it_has_bias = True
        
    # Output Shape
    output_dim = -1
    output_shape_string = "{-1, -1}"
    if (layer_output != None):
        output_dim = len(layer_output.shape)
        # HARDCODED: reversed order. From [1, 5, 1600] to [1600, 5, 1]
        output_shape_string = "{"
        for x in range(output_dim - 1, 0 - 1, -1):
            output_shape_string += str(layer_output.shape[x])
            if (x > 0):
                output_shape_string += ", "
            else:
                output_shape_string += "}"
        
    # Weights Data and Shape
    weight = module.weight.detach().cpu().numpy()
    weight_row = weight.shape[0]
    weight_col = weight.shape[1]
    
    string_weight_shape = "{" + str(weight_row) + ", " + str(weight_col) + "}"
    weight_dim = len(weight.shape)
    string_weight_dim = str(weight_dim)
    
    
    # -------------- check if weight are sparse -------------- #
    non_zero_values = np.count_nonzero(weight)
    sparse = False
    size_requested = weight_row * weight_col
    
    # check representation
    if representation == 'pair':
        sparse = non_zero_values < ((weight_row * weight_col) / 2)
        size_requested = non_zero_values * 2
    elif representation == 'csc':
        sparse = non_zero_values < ((weight_col * (weight_row - 1) - 1) / 2)
        size_requested = (non_zero_values * 2) + weight_col + 1
    elif representation == 'csr':
        sparse = non_zero_values < ((weight_row * (weight_col - 1) - 1) / 2)
        size_requested = (non_zero_values * 2) + weight_row + 1
    sparse_string = "DENSE"
    
    # used in both but could be edited if sparse
    string_weight_addr = layer_name + "_w"
    string_strides = "{" + str(weight_col) + ", " + str(1) + "}"
    # used only if sparse
    string_sparse_weight_shape = ""
    string_sparse_weight_dim = ""
    string_sparse_offset_addr = ""
    string_sparse_sizes_addr = ""
    
    if (sparse):
        sparse_string = "SPARSE"
        string_sparse_weight_shape = string_weight_shape
        string_sparse_weight_dim = string_weight_dim
        param_nnz_values = layer_name.upper() + "_NNZ_LEN"
        
        # for the weights declaration
        string_weight_shape = "{" + param_nnz_values + "}"
        string_weight_dim = "1"
        
        string_weight_addr = layer_name + "_w_data"
        string_strides = "{1}"
        string_sparse_offset_addr = layer_name + "_w_offset"
        string_sparse_sizes_addr = layer_name + "_w_sizes"
        
        # ----------------- WRITING TO THE FILE --------------------#
        # print macro
        print("\n#define " + param_nnz_values + f' {non_zero_values}', file=f)
        print("\n// Representation of the Sparse Matrix: " + representation + "\n", file=f)
        
        if representation == 'pair':
            weight_values = np.empty([non_zero_values], dtype=float)
            weight_index = np.empty([non_zero_values], dtype=int)

            weight = weight.ravel()

            # returns index of non zero values of flattened version array
            index_nnz_list = np.flatnonzero(weight)

            assert index_nnz_list.size == non_zero_values, "Non Zero Values Count Error"
            for index, index_value in enumerate(index_nnz_list):
                weight_index[index] = index_value
                weight_values[index] = weight[index_value]
            assert weight_index.size == weight_values.size, "Array should be same length"

            # Print the weight matrix as value-index pair

            # values
            if (flit):
                print(f'__ro_hifram fixed {string_weight_addr}[{non_zero_values}] = {{', file=f)
            else:
                print(f'float {string_weight_addr}_data[{non_zero_values}] = {{', file=f)
            with np.printoptions(floatmode='maxprec'):
                print("\t", end='', file=f)
                if (flit):
                    print("F_LIT(", end='', file=f)
                    print(*weight_values, sep='), F_LIT(', end='', file=f)
                    print(")", end='', file=f)
                else:
                    print(*weight_values, sep=', ', file=f)
            print('};\n', file=f)

            # index
            if (flit):
                print(f'__ro_hifram uint16_t {layer_name}_index[{non_zero_values}] = {{', file=f)
            else:
                print(f'uint16_t {layer_name}_index[{non_zero_values}] = {{', file=f)
            print("\t", end='', file=f)
            print(*weight_index, sep=', ', file=f)
            print('};\n', file=f)

        elif representation == 'csc':
            # Convert the weight matrix to the CSC format
            weight_csc = sp.csc_matrix(weight)

            # Print the weight matrix as CSC data arrays
            if (flit):
                print(f'__ro_hifram fixed {string_weight_addr}[{non_zero_values}] = {{', file=f)
            else:
                print(f'float {string_weight_addr}[{non_zero_values}] = {{', file=f)
            with np.printoptions(floatmode='maxprec'):
                print("\t", end='', file=f)
                if (flit):
                    print("F_LIT(", end='', file=f)
                    print(*weight_csc.data, sep='), F_LIT(', end='', file=f)
                    print(")", end='', file=f)
                else:
                    print(*weight_csc.data, sep=', ', file=f)
            print('};\n', file=f)

            if (flit):
                print(f'__ro_hifram uint16_t {string_sparse_offset_addr}[{non_zero_values}] = {{', file=f)
            else:
                print(f'uint16_t {string_sparse_offset_addr}[{non_zero_values}] = {{', file=f)
            print("\t", end='', file=f)
            print(*weight_csc.indices, sep=', ', file=f)
            print('};\n', file=f)

            if (flit):
                print(f'__ro_hifram uint16_t {string_sparse_sizes_addr}[{weight_col+1}] = {{', file=f)
            else:
                print(f'uint16_t {string_sparse_sizes_addr}[{weight_col+1}] = {{', file=f)
            print("\t", end='', file=f)
            print(*weight_csc.indptr, sep=', ', file=f)
            print('};\n', file=f)
        elif representation == 'csr':
            # Convert the weight matrix to the CSC format
            weight_csr = sp.csr_matrix(weight)

            # Print the weight matrix as CSC data arrays
            if (flit):
                print(f'__ro_hifram fixed {string_weight_addr}[{non_zero_values}] = {{', file=f)
            else:
                print(f'float {string_weight_addr}[{non_zero_values}] = {{', file=f)
            with np.printoptions(floatmode='maxprec'):
                print("\t", end='', file=f)
                if (flit):
                    print("F_LIT(", end='', file=f)
                    print(*weight_csr.data, sep='), F_LIT(', end='', file=f)
                    print(")", end='', file=f)
                else:
                    print(*weight_csr.data, sep=', ', file=f)
            print('};\n', file=f)

            if (flit):
                print(f'__ro_hifram uint16_t {string_sparse_offset_addr}[{non_zero_values}] = {{', file=f)
            else:
                print(f'uint16_t {string_sparse_offset_addr}[{non_zero_values}] = {{', file=f)
            print("\t", end='', file=f)
            print(*weight_csr.indices, sep=', ', file=f)
            print('};\n', file=f)

            if (flit):
                print(f'__ro_hifram uint16_t {string_sparse_sizes_addr}[{weight_row+1}] = {{', file=f)
            else:
                print(f'uint16_t {string_sparse_sizes_addr}[{weight_row+1}] = {{', file=f)
            print("\t", end='', file=f)
            print(*weight_csr.indptr, sep=', ', file=f)
            print('};\n', file=f)
        
        # ----------------- END PRINTING TO FILE -------------------#
    else:
        # Print Weight array
        print('\n', file=f)
        if (flit):
            print("__ro_hifram fixed " + string_weight_addr + f'[{weight_row}][{weight_col}]' + " = {", file=f)
        else:
            print("float " + string_weight_addr + f'[{weight_row}][{weight_col}]' + " = {", file=f)
        print("\t", end='', file=f)
        with np.printoptions(floatmode='maxprec'):
            if (flit):
                print("F_LIT(", end='', file=f)
                print(*weight.ravel(), sep='), F_LIT(', end='', file=f)
                print(")", end='', file=f)
            else:
                print(*weight.ravel(), sep=', ', file=f)
        print("\n};\n", file=f)
        
    # ------------- BIAS PRINT ------------- #
    string_bias_strides = "{1, 1}"
    string_bias_addr = layer_name + "_b"
    string_bias_shape = ""
    string_bias_dim = ""
    if (it_has_bias):
        # bias data and shape HARDCODED as reversed and (value, 1)
        bias = module.bias.detach().cpu().numpy()
        bias_row = bias.shape[0]

        string_bias_shape = "{" + str(bias_row) + ", " + str(1) + "}"
        bias_dim = 2
        string_bias_dim = str(bias_dim)
        
        # Print Bias array
        if (flit):
            print("__ro_hifram fixed " + string_bias_addr + "["+ str(bias_row) +"] = {", file=f)
        else:
            print("float " + string_bias_addr + "["+ str(bias_row) +"] = {", file=f)
        print("\t", end='', file=f)
        with np.printoptions(floatmode='maxprec'):
            if (flit):
                print("F_LIT(", end='', file=f)
                print(*bias.ravel(), sep='), F_LIT(', end='', file=f)
                print(")", end='', file=f)
            else:
                print(*bias.ravel(), sep=', ', file=f)
        print("\n};\n", file=f)
    
    intro = "/* Fully Connected layer " + str(num_this_layer_type) + " parameters */\n"
    
    # --------- weights parameters ---------- #
    tensor_weight_declaration = "__fram tensor " + weight_name + " = {\n"
    tensor_weight_declaration += "\t.shape = " + string_weight_shape + ",\n"
    tensor_weight_declaration += "\t.dimension = " + string_weight_dim + ",\n"
    tensor_weight_declaration += "\t.strides = " + string_strides + ",\n"
    tensor_weight_declaration += "\t.dataAddr = " + "&" + string_weight_addr + ",\n"
    
    # if the weights are sparse
    if (sparse):
        tensor_weight_declaration += "\t.sparse = {\n"
        tensor_weight_declaration += "\t\t.offset = " + "&" + string_sparse_offset_addr + ",\n"
        tensor_weight_declaration += "\t\t.size = " + "&" + string_sparse_sizes_addr + ",\n"
        tensor_weight_declaration += "\t\t.dimension = " + string_sparse_weight_dim + ",\n"
        tensor_weight_declaration += "\t\t.shape = " + string_sparse_weight_shape + ",\n"
        tensor_weight_declaration += "\t},\n"
    
    tensor_weight_declaration += "};\n\n"
    
    # --------- if there is a bias --------- #
    if (it_has_bias):
        tensor_bias_declaration = "__fram tensor " + bias_name + " = {\n"
        tensor_bias_declaration += "\t.shape = " + string_bias_shape + ",\n"
        tensor_bias_declaration += "\t.dimension = " + string_bias_dim + ",\n"
        tensor_bias_declaration += "\t.strides = " + string_bias_strides + ",\n"
        tensor_bias_declaration += "\t.dataAddr = " + "&" + string_bias_addr + ",\n"
        tensor_bias_declaration += "};\n\n"
    
    # ---------- layer parameters ---------- #
    fully_con_t_declaration = "__fram fully_con_t " + param_name + " = {\n"
    fully_con_t_declaration += "\t.activation = " + next_activation + ",\n"
    fully_con_t_declaration += "\t.sparseness = " + sparse_string + ",\n"
    fully_con_t_declaration += "\t.weights = &" + weight_name + ",\n"
    if (it_has_bias):
        fully_con_t_declaration += "\t.bias = &" + bias_name + ",\n"
    fully_con_t_declaration += "\t.outputShape = " + output_shape_string + ",\n"
    fully_con_t_declaration += "\t.outputDimension = " + str(output_dim) + ",\n"
    fully_con_t_declaration += "};\n\n"
    
    string_result = intro + tensor_weight_declaration
    if (it_has_bias):
        string_result += tensor_bias_declaration
    string_result += fully_con_t_declaration
    return string_result

def conv_declaration(module, param_name, num_this_layer_type, layer_output, next_activation, representation, layer_name, f, flit):
    weight_name = "weights_Convol_2D_" + str(num_this_layer_type)
    bias_name = "bias_Convol_2D_" + str(num_this_layer_type)
    #param_name = "param_Conv2D_" + str(num_this_layer_type)
    
    # Check if there is a bias
    it_has_bias = False
    if (hasattr(module, 'bias') and module.bias is not None):
        it_has_bias = True
        
    # Output Shape
    output_dim = -1
    output_shape_string = "{-1, -1}"
    if (layer_output != None):
        output_dim = len(layer_output.shape)
        # HARDCODED: right order. From [20, 24, 24] to [24, 20, 20]
        output_shape_string = "{"
        for x in range(1, output_dim):
            output_shape_string += str(layer_output.shape[x])
            if (x < output_dim - 1):
                output_shape_string += ", "
            else:
                output_shape_string += "}"
                
    # Stride is INT or a list - HARDCODED TO BE THREE
    stride = module.stride
    stride_list = []
    if isinstance(stride, int):
        stride_list.append(stride)
        stride_list.append(stride)
        stride_list.append(stride)
    else:
        for x in range(0, len(stride)):
            stride_list.append(stride[x])
    while (len(stride_list) < 3):
        stride_list.append(stride[0])
    strideString = "{" + str(stride_list[0]) + ", " + str(stride_list[1]) + ", " + str(stride_list[2]) + "}"
        
    # Check padding - NOT IMPLEMENTED
    paddingString = 'PAD_NONE'
    
                
    # Weights Data and Shape
    weight = module.weight.detach().cpu().numpy()
    n_filters = weight.shape[0]
    n_depth = weight.shape[1]
    n_height = weight.shape[2]
    n_width = weight.shape[3]
    
    string_weight_shape = "{" + f'{n_filters}, {n_depth}, {n_height}, {n_width}' + "}"
    weight_dim = len(weight.shape)
    string_weight_dim = str(weight_dim)
    
    # -------------- check if weight are sparse -------------- #
    non_zero_values = np.count_nonzero(weight)
    sparse = False
    size_requested = n_filters * n_depth * n_height * n_width
    if representation == 'pair':
        sparse = non_zero_values < (size_requested / 2)
        size_requested = non_zero_values * 2
    elif representation == 'csc' or representation == 'csr':
        sparse = non_zero_values < ((size_requested - n_filters) / 2)
        size_requested = (non_zero_values * 2) + n_filters
        
    # ------------------ HARDCODED TO BE SPARSE ------------------- #
    sparse = True

    # used in both but could be edited if sparse
    string_weight_addr = layer_name + "_w"
    string_strides = "{1}"
    # used only if sparse
    string_sparse_weight_shape = ""
    string_sparse_weight_dim = ""
    string_sparse_offset_addr = ""
    string_sparse_sizes_addr = ""
    
    if (sparse):
        sparse_string = "SPARSE"
        string_sparse_weight_shape = string_weight_shape
        string_sparse_weight_dim = string_weight_dim
        param_nnz_values = layer_name.upper() + "_NNZ_LEN"
        
        # for the weights declaration
        string_weight_shape = "{" + param_nnz_values + "}"
        string_weight_dim = "1"
        
        string_weight_addr = layer_name + "_w_data"
        string_sparse_offset_addr = layer_name + "_w_offset"
        string_sparse_sizes_addr = layer_name + "_w_sizes"
        
        # ----------------- WRITING TO THE FILE --------------------#
        # print macro
        print("\n#define " + param_nnz_values + f' {non_zero_values}', file=f)
        print("\n// Representation of the Sparse Matrix: " + representation + "\n", file=f)
       
        if (representation == 'csc' or representation == 'csr'):
            # CSC or CSR format
            filters_values = np.empty([non_zero_values], dtype=float)
            filters_offsets = np.empty([non_zero_values], dtype=int)
            filters_sizes = np.empty([n_filters], dtype=int)

            count = 0
            value_position = 0

            for index, filter_weight in enumerate(weight): # from 0 to number_filters

                # insert filters non zero values into the fitler sizes
                non_zero_values_here = np.count_nonzero(filter_weight)
                filters_sizes[index] = non_zero_values_here
                # flatten the filter
                flatten_filter = filter_weight.ravel()
                offset = 1
                for value in flatten_filter: # from 0 to n_depth * n_height * n_width
                    if (value == 0):
                        # increase offset
                        offset += 1
                    else:
                        # save value, save index, reset offset, increase position
                        filters_values[value_position] = value
                        filters_offsets[value_position] = offset
                        offset = 1
                        value_position += 1

            # Print Values array
            if (flit):
                print("__ro_hifram fixed " + string_weight_addr + "["+ str(non_zero_values) +"] = {", file=f)
            else:
                print("float " + string_weight_addr + "["+ str(non_zero_values) +"] = {", file=f)
            print("\t", end='', file=f)
            with np.printoptions(floatmode='maxprec'):
                if (flit):
                    print("F_LIT(", end='', file=f)
                    print(*filters_values, sep='), F_LIT(', end='', file=f)
                    print(")", end='', file=f)
                else:
                    print(*filters_values, sep=', ', file=f)
            print("\n};\n", file=f)

            # Print Offsets array
            if (flit):
                print("__ro_hifram uint16_t " + string_sparse_offset_addr + "["+ str(non_zero_values) +"] = {", file=f)
            else:
                print("uint16_t " + string_sparse_offset_addr + "["+ str(non_zero_values) +"] = {", file=f)
            print("\t", end='', file=f)
            print(*filters_offsets, sep=', ', file=f)
            print("\n};\n", file=f)

            # Print Sizez array
            if (flit):
                print("__ro_hifram uint16_t " + string_sparse_sizes_addr + "["+ str(n_filters) +"] = {", file=f)
            else:
                print("uint16_t " + string_sparse_sizes_addr + "["+ str(n_filters) +"] = {", file=f)
            print("\t", end='', file=f)
            print(*filters_sizes, sep=', ', file=f)
            print("\n};\n", file=f)
    
    # ------------- BIAS PRINT ------------- #
    string_bias_strides = "{1}"
    string_bias_addr = layer_name + "_b"
    string_bias_shape = ""
    string_bias_dim = ""
    if (it_has_bias):
        # bias data and shape HARDCODED as reversed and (value, 1)
        bias = module.bias.detach().cpu().numpy()
        bias_row = bias.shape[0]

        string_bias_shape = "{" + str(bias_row) + "}"
        bias_dim = 1
        string_bias_dim = str(bias_dim)
        
        # Print Bias array
        if (flit):
            print("__ro_hifram fixed " + string_bias_addr + "["+ str(bias_row) +"] = {", file=f)
        else:
            print("float " + string_bias_addr + "["+ str(bias_row) +"] = {", file=f)
        print("\t", end='', file=f)
        with np.printoptions(floatmode='maxprec'):
            if (flit):
                print("F_LIT(", end='', file=f)
                print(*bias.ravel(), sep='), F_LIT(', end='', file=f)
                print(")", end='', file=f)
            else:
                print(*bias.ravel(), sep=', ', file=f)
        print("\n};\n", file=f)
    
    intro = "/* Convolutional layer " + str(num_this_layer_type) + " parameters */\n"
    
    # --------- weights parameters ---------- #
    tensor_weight_declaration = "__fram tensor " + weight_name + " = {\n"
    tensor_weight_declaration += "\t.shape = " + string_weight_shape + ",\n"
    tensor_weight_declaration += "\t.dimension = " + string_weight_dim + ",\n"
    tensor_weight_declaration += "\t.strides = " + string_strides + ",\n"
    tensor_weight_declaration += "\t.dataAddr = " + "&" + string_weight_addr + ",\n"
    
    # if the weights are sparse
    if (sparse):
        tensor_weight_declaration += "\t.sparse = {\n"
        tensor_weight_declaration += "\t\t.offset = " + "&" + string_sparse_offset_addr + ",\n"
        tensor_weight_declaration += "\t\t.size = " + "&" + string_sparse_sizes_addr + ",\n"
        tensor_weight_declaration += "\t\t.dimension = " + string_sparse_weight_dim + ",\n"
        tensor_weight_declaration += "\t\t.shape = " + string_sparse_weight_shape + ",\n"
        tensor_weight_declaration += "\t},\n"
    
    tensor_weight_declaration += "};\n\n"
    
    if (it_has_bias):
        # --------- if there is a bias --------- #
        tensor_bias_declaration = "__fram tensor " + bias_name + " = {\n"
        tensor_bias_declaration += "\t.shape = " + string_bias_shape + ",\n"
        tensor_bias_declaration += "\t.dimension = " + string_bias_dim + ",\n"
        tensor_bias_declaration += "\t.strides = " + string_bias_strides + ",\n"
        tensor_bias_declaration += "\t.dataAddr = " + "&" + string_bias_addr + ",\n"
        tensor_bias_declaration += "};\n\n"
        
        # --------- printing bias -------------- #
    
    # ---------- layer parameters ---------- #
    conv_t_declaration = "__fram conv_t " + param_name + " = {\n"
    conv_t_declaration += "\t.stride = " + strideString + ",\n"
    conv_t_declaration += "\t.padding = " + paddingString + ",\n"
    conv_t_declaration += "\t.weights = &" + weight_name + ",\n"
    if (it_has_bias):
        conv_t_declaration += "\t.bias = &" + bias_name + ",\n"
    conv_t_declaration += "\t.outputDimension = " + str(output_dim - 1) + ",\n"
    conv_t_declaration += "\t.outputShape = " + output_shape_string + ",\n"
    conv_t_declaration += "\t.type = " + 'NORMAL' + ",\n"
    conv_t_declaration += "\t.activation = " + next_activation + ",\n"
    conv_t_declaration += "};\n\n"
    
    string_result = intro + tensor_weight_declaration
    if (it_has_bias):
        string_result += tensor_bias_declaration
    string_result += conv_t_declaration
    return string_result

'''
    Input still hardcoded
'''
def input_declaration():
    param_name = "input_data"
    
    intro = "/* Input data parameters */\n"
    dataString = "&input"
    dimString = "3"
    shapeString = "{1, 28, 28}"
    strideString = "{784, 28, 1}"
    
    # ---------- layer parameters ---------- #
    tensor_declaration = "__fram tensor " + param_name + " = {\n"
    tensor_declaration += "\t.shape = " + shapeString + ",\n"
    tensor_declaration += "\t.dimension = " + dimString + ",\n"
    tensor_declaration += "\t.strides = " + strideString + ",\n"
    tensor_declaration += "\t.dataAddr = " + dataString + ",\n"
    tensor_declaration += "};\n\n"
    
    return intro + tensor_declaration

''' Given the layer dictionary return the number of instances of the layer type
    It check if the layer type is already present, if not it adds it
    starting value: 1, otherwise add 1 to the corresponding number of instances
'''
def update_layer(dictionary, key):
    if (key in dictionary):
        dictionary[key] += 1
    else:
        dictionary[key] = 1
    return dictionary[key]

''' Recursive method to obtain the full list of module in topological order
    is depth-first (it should be the way the same order of the forward function)
    "Unwrap" the sequential layer of the model
'''
def create_module_list(model):
    module_list = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # If it's a sequential container, recursively get its layers
            module_list.extend(create_module_list(module))
        else:
            module_list.append((name, module))
    return module_list

def save_compressed_model(model, representation, input_data=None, directory="my_model", flit=False, verbose=False):
    assert isinstance(model, nn.Module), "The model is not a subclass of torch.nn.Module"
    assert representation in ['pair', 'csc', 'csr'], "Representation Invalid"
    assert representation != 'pair', "Not supported with the model.h"
    
    # Path
    path = directory

    # Create the directory
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory '% s' created" % directory)
        else:
            print("Directiory already existing") 
    except:
      print("Something went wrong")
    
    # changing the path for the file -> path is now = my_model/
    path = directory + "/"
    
    # We need to create a headers directory, a model.c and model.h files
    # -> path_headers is now = my_model/headers
    path_headers = path + "headers"
    # Create the directory
    try:
        if not os.path.exists(path_headers):
            os.makedirs(path_headers)
            print("Directory '% s' created" % path_headers)
        else:
            print("Directiory already existing") 
    except:
      print("Something went wrong")
    
    # changing the path_headers for the files -> path_headers is now = my_model/headers/
    path_headers = path_headers + "/"
    
    # Unwrap all the sequential layer in order
    unwrapped_named_children = create_module_list(model)
    
    # --------------- VARIABLES FOR THE CONVERSION ------------ #
    layers_declaration_string = ""
    layers_definition_string = "void loadModel(){\n\n"
    total_number_layers = 1
    layers_files_names_list = []
    layer_types_num = {}
    
    # To perform forward operation to track the shape of the input through the model
    layer_input = None
    layer_output = input_data
    
    layer_number = 1
    
    # -------------------- Add Input Section: it is hardcoded ------------------ #
    param_name = "input_data"
    layers_declaration_string += input_declaration()
    layers_definition_string += "\t// Input Layer\n"
    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = INPUT;\n"
    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"
    layer_number += 1
    verbose and print("\t\tSaved: " + param_name)
    with open(path_headers + 'input.h', 'w') as f:
        # create new file for each Layer (name of the module)
        verbose and print("\t\tWorking in: " + file_name)
        # Printing Header
        print("#ifndef INPUT_H", file=f)
        print("#define INPUT_H", file=f)
        # FLIT
        if (flit):
            #include <includes/memlib/mem.h>
            #include <includes/fixedlib/fixed.h>
            print("#include <includes/memlib/mem.h>", file=f)
            print("#include <includes/fixedlib/fixed.h>\n", file=f)
        # ----------- printing input -------- #
        if (flit):
            print("__ro_hifram fixed input[1][28][28] = {", file=f)
        else:
            print("__ro_fram float input[1][28][28] = {", file=f)
        print("\t", end='', file=f)
        for i in range(number_of_examples):
            verbose and print(example_data[i])
            with np.printoptions(floatmode='maxprec'):
                if (flit):
                    print("F_LIT(", end='', file=f)
                    print(*example_data[i].ravel(), sep='), F_LIT(', end='', file=f)
                    print(")", end='', file=f)
                else:
                    print(*example_data[i].ravel(), sep=', ', end='', file=f)
                if (i != number_of_examples - 1):
                    print(",\n\n\t", end='', file=f)
        print("\n};\n", file=f)
        if (flit):
           print(f'__ro_hifram fixed labels[{number_of_examples}' + "] = {", file=f)
        else:
            print(f'__ro_fram int labels[{number_of_examples}' + "] = {", file=f)
        print("\t", end='', file=f)
        for i in range(number_of_examples):
            verbose and print(example_targets[i])
            print(example_targets[i], end='', file=f)
            if (i != number_of_examples - 1):
                    print(", ", end='', file=f)
        print("\n};", file=f)
        # ---------- input printed -------------- #
        outputs = model.to('cpu').forward(example_data_tensor[0].unsqueeze(0).to('cpu'))
        _, predicted = torch.max(outputs.data, 1)
        print(f'// Model prediction: {predicted.item()}', file=f)
                                                              
        # print final header
        print("\n#endif", file=f)
    layers_files_names_list.append("input.h")
    # ---------------- END INPUT ------------------------------- #
    
    # ---------------- ITERATING OVER THE MODULES -------------- #
    print("\nIteratiting over the module:\n")
    for index, (name, module) in enumerate(unwrapped_named_children):
        verbose and print("\t",name, module, str(module))
        
        # check type of module
        is_conv2d = isinstance(module, torch.nn.Conv2d)
        is_linear = isinstance(module, torch.nn.Linear)
        is_maxpool = isinstance(module, torch.nn.MaxPool2d)
        is_softmax = isinstance(module, torch.nn.LogSoftmax)
        is_flatten = isinstance(module, torch.nn.Flatten)
        
        type = ""
        if (is_conv2d):
            layer_type = "conv"
        elif (is_linear):
            layer_type = "fc"
        elif(is_maxpool):
            layer_type = "pooling"
        elif(is_softmax):
            layer_type = "softmax"
        elif(is_flatten):
            layer_type = "flatten"
        else:
            assert isinstance(module, nn.ReLU), "Layer not recognized"
            layer_type = "relu"
        # get the name of a module: type(module).__name__
        
        # update the corresponding layer number
        num_type = update_layer(layer_types_num, layer_type)
        
        # check if the following one is a RELU layer
        next_activation = "ACT_NONE"
        if index + 1 < len(unwrapped_named_children) and isinstance(unwrapped_named_children[index + 1][1], nn.ReLU):
            next_activation = "RELU"
        
        # Get the previous layer output and do a forward operation
        layer_input = layer_output
        if (layer_input != None):
            layer_output = module(layer_input)
            if (hasattr(module, 'weight') and module.weight is not None):
                print("Weight shape:", module.weight.shape)
            print("Shape output of the module:", layer_output.shape)
        
        # I need to save the parameters which I need for the inference
        # only the module containing parameters are saved as a separate file
        
        ''' Based on the type of the layer I perform different operation:
                - if has parameters I create a new file to print them
                    - add the name to the layers_files_names_list
                - add to the model declaration string
                - add to the model definition string
            Layers skipped:
                - ReLU
        '''
        
        # ------------ LAYERS THAT DO NOT REQUIRE A NEW FILES -------- #
        if (is_maxpool):
            '''Maxpool's shared variables between different section(string):
                    - name of pooling_t: param_maxPool_#(num_type)
            '''
            param_name = "param_maxPool_" + str(num_type)
            layers_declaration_string += pooling_declaration(module, param_name, num_type, layer_output)
            layers_definition_string += "\t// Max Pool: " + str(num_type) + "\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = MAXPOOL;\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"
            
            layer_number += 1
            verbose and print("\t\tSaved: " + param_name)
            
        elif (is_softmax):
            '''LogSoftmax's shared variables between different section(string):
                    - name of softmax_t: param_softmax_#(num_type)
            '''
            param_name = "param_softmax_" + str(num_type)
            layers_declaration_string += softmax_declaration(module, param_name, num_type, layer_input)
            layers_definition_string += "\t// Softmax layer: " + str(num_type) + "\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = SOFTMAX;\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"
                    
            layer_number += 1
            verbose and print("\t\tSaved: " + param_name)
            
        elif (is_flatten):
            '''FLatten's shared variables between different section(string):
                    - name of reshape_t: param_flatten_#(num_type)
            '''
            param_name = "param_flatten_" + str(num_type)
            layers_declaration_string += flatten_declaration(module, param_name, num_type, layer_output)
            layers_definition_string += "\t// Flatten: " + str(num_type) + "\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = FLATTEN;\n"
            layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"
            
            layer_number += 1
            verbose and print("\t\tSaved: " + param_name)
            
        # ------------ LAYERS THAT DO REQUIRE A NEW FILES ----------- #
        if (is_conv2d or is_linear):
            
            # i.e., conv1, CONV1_H, my_model/headers/conv1.h
            layer_name = layer_type + str(num_type)
            header_name = layer_name.upper() + "_H"
            file_name = path_headers + layer_name + ".h"
            
            with open(file_name, 'w') as f:
                # create new file for each Layer (name of the module)
                verbose and print("\t\tWorking in: " + file_name)

                # Printing Header
                print("#ifndef " + header_name, file=f)
                print("#define " + header_name, file=f)
                print("// Header file of the " + str(layer_number) 
                      + "-th layer of the model - The type of the layer is " + layer_type, file=f)
                
                # FLIT
                if (flit):
                    #include <includes/memlib/mem.h>
                    #include <includes/fixedlib/fixed.h>
                    print("#include <includes/memlib/mem.h>", file=f)
                    print("#include <includes/fixedlib/fixed.h>", file=f)
                
                if (is_conv2d):
                    ''' Cond2d's shared variables between different section(string):
                            - name of conv_t: param_Conv2D_#(num_type)
                        Conv2d's shared variables between declaration string and files:
                            - possible MACROS: CONV#_WMD_LEN, CONV#_WMH_LEN, CONV#_WMV_LEN
                            - dataAddr: conv#_wmd
                            - offset: conv#_wmd_offsets
                            - size: conv#_wmd_sizes
                            - possible bias dataAddr: conv#_b
                    '''
                    param_name = "param_Conv2D_" + str(num_type)
                    layers_declaration_string += conv_declaration(module, param_name, num_type, layer_output, next_activation, representation, layer_name, f, flit)
                    layers_definition_string += "\t// Conv" + str(num_type) + " layer\n"
                    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = CONV2D;\n"
                    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"

                    

                elif (is_linear):
                    ''' Linear's shared variables between different section(string):
                            - name of fully_con_t: param_fully_con_#(num_type)
                        Linear's shared variables between declaration string and files:
                            - possible MACROS: FC#_WMH_LEN, FC#_WMV_LEN
                            - dataAddr: fc#_wmh or fc#_wmv (sparse), fc#_w (dense)
                            - offset: fc#_wmh_offsets or fc#_wmv_offsets (sparse)
                            - size: fc#_wmh_sizes or fc#_wmv_sizes (sparse)
                            - possible bias dataAddr: fc#_b
                    '''
                    param_name = "param_fully_con_" + str(num_type)
                    layers_declaration_string += linear_declaration(module, param_name, num_type, layer_output, next_activation, representation, layer_name, f, flit)
                    layers_definition_string += "\t// FC" + str(num_type) + " layer\n"
                    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].type = FULLYCONNECTED;\n"
                    layers_definition_string += "\tModel[" + str(layer_number - 1) + "].ptr = (void *)&" + param_name + ";\n"
                    
                    


                # print final header
                print("\n#endif", file=f)
                
                # ------------ FINISHED WRITING IN THE FILE ------------  #
            
            # add the file to the list of files to include
            layers_files_names_list.append(layer_name + ".h")
            
            # increasing layer_number
            layer_number += 1
            
            # --------- FINISHED IF CONV/ LINEAR LAYER --------- #
        
        total_number_layers += 1
        print("Layer", layer_type, num_type, "Values:", layer_number - 1, total_number_layers - 1)
        # --------- FINISHED CURRENT LAYER --------- #
    
    # -----------FINISHED ITERATING THROUGH THE MODULES --------- #
    print("Checked " + str(total_number_layers - 1) + " / " + str(len(unwrapped_named_children)) + " layers\n")
    print("Saved " + str(layer_number - 1) + " / " + str(len(unwrapped_named_children)) + " layers\n")
    
    # creation of the headers file
    header_string = "#ifndef HEADERS_HEADERS_H_\n#define HEADERS_HEADERS_H_\n\n"
    header_string += "/* Input and Weights Headers */\n"
    for file in layers_files_names_list:
        header_string += "#include \"headers/" + file + "\"\n"
    header_string += "\n\n\n#endif /* HEADERS_HEADERS_H_ */"
    
    file_name = path_headers + "headers.h"
    with open(file_name, 'w') as f:
        # Printing Header
        print(header_string, file=f)
    
    # creation of the model.c file
    model_c_string = "#include \"ml.h\"\n"
    model_c_string += "#include \"headers/headers.h\" // header files which contain weights and bias\n\n"
    model_c_string += "#define NUM_MODEL_LAYERS " + str(layer_number - 1) + "\n\n"
    model_c_string += "__fram sequentialModel Model[NUM_MODEL_LAYERS];\n\n"
    
    model_c_string += layers_declaration_string + "\n"
    
    model_c_string += layers_definition_string + "\n}"
    
    file_name = path + "model.c"
    with open(file_name, 'w') as f:
        # Printing Header
        print(model_c_string, file=f)
    
    # creation of the model.h file
    model_h_string = "#ifndef MODEL_MODEL_H_\n#define MODEL_MODEL_H_\n\n"
    model_h_string += "#define NUM_MODEL_LAYERS " + str(layer_number - 1) + "\n\n"
    model_h_string += "extern sequentialModel Model[];\n\n"
    model_h_string += "void loadModel();\n"
    model_h_string += "\n\n#endif /* MODEL_MODEL_H_ */"
    
    file_name = path + "model.h"
    with open(file_name, 'w') as f:
        # Printing Header
        print(model_h_string, file=f)
    
    print("Finished Saving the Model")

# save_compressed_model(model, 'csc')