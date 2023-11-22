import typer
import mlflow
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from simplify_leaves_mnist import FFFWrapper
from matplotlib import pyplot as plt


def get_split_code(array, bias):
    code = """
    acc = """ + " + ".join(f"x[{i}] * {v}" for i, v in enumerate(array)) + """;
    acc += """ + str(bias[0]) + """;
    """
    return code


def get_output_code(w1, b1, w2, b2):
    code = """
    float hidden[""" + str(w1.shape[1]) + """];
    """
    for i in range(w1.shape[1]):
        code += f"hidden[{i}] = {b1[i]} + " + " + ".join(f"x[{j}] * {v}" for j, v in enumerate(w1[:, i])) + ";\n"
        code += f"hidden[{i}] = hidden[{i}] > 0 ? hidden[{i}] : 0;\n"
    code += """
    float logits[""" + str(w2.shape[1]) + """];
    """
    for j in range(w2.shape[1]):
        code += f"logits[{j}] = {b2[j]} + " + " + ".join(f"hidden[{i}] * {v}" for i, v in enumerate(w2[:, j])) + ";\n"
        code += f"logits[{j}] = logits[{j}] > 0 ? logits[{j}] : 0;\n"

    code += """
    max = 0.0;
    argmax = 0;
    for (int i = 0; i < """ + str(w2.shape[1]) + """; i++) {
        if (logits[i] > max) {
            max = logits[i];
            argmax = i;
        }
    }

    return argmax;
    """

    return code

def get_splits(weights, biases):
    code = """int perform_inference(float* x) {
    float acc;
    float max;
    int argmax;
        <replaceme>
}"""
    for index, (array, bias) in enumerate(zip(weights, biases)):
        code = code.replace("<replaceme>", get_split_code(array, bias), 1)
    return code


class Node:
    def __init__(self, array, bias, left, right):
        self._array = array
        self._bias = bias
        self._left = left
        self._right = right

    def __str__(self):
        code = get_split_code(self._array, self._bias)

        code += """
        if (acc >= 0) {
            """ + str(self._left) + """
        } else {
            """ + str(self._right) + """
        }
        """
        return code


class Leaf(Node):
    def __init__(self, w1, b1, w2, b2):
        self._w1 = w1
        self._b1 = b1
        self._w2 = w2
        self._b2 = b2

    def __str__(self):
        if self._w2 is None:
            return f"return {self._w1};\n"
        return get_output_code(self._w1, self._b1, self._w2, self._b2)

def print_parameters(params, key, lines, line, flit):
    # TODO: the first check on the sparsity on the weight consider the non zero values of truncated leaves
    # it should count the nnz of only kept leaves
    
    weight = params[key]
    dim = len(weight.shape)
    
    non_zero_values = np.count_nonzero(weight)
    first_dim = weight.shape[0]
    sparse = False
    
    if (dim == 2):
        weight_row = weight.shape[0]
        weight_col = weight.shape[1]
        
        sparse = non_zero_values < ((weight_row * (weight_col - 1) - 1) / 2)
    elif (dim == 3):
        weight_depth = weight.shape[0]
        weight_row = weight.shape[1]
        weight_col = weight.shape[2]
        
        size_requested = weight_depth * weight_row * weight_col
        sparse = non_zero_values < ((size_requested - weight_depth) / 2)
    if (not sparse):
        # print the parameters as usal
        lines.insert(
            line,
            "#define " + key + "_SPARSE " + str(0) + "\n"
        )
        line += 1
        line += 1
        
        param = weight
        if (key in ['LEAF_HIDDEN_WEIGHTS', 'LEAF_HIDDEN_BIASES', 'LEAF_OUTPUT_WEIGHTS', 'LEAF_OUTPUT_BIASES']):
            param = param[params['FASTINFERENCE'] == -1]
        param = param.flatten()
        tmp = ""
        if (flit and key not in ['FASTINFERENCE']):
            tmp = ", ".join(["F_LIT(" + str(x) + ")" for x in param])
        else:
            tmp = ", ".join([str(x) for x in param])
        lines.insert(
            line,
            tmp
        )
    else:
        # CSC or CSR format
        leaves_values = np.empty([0], dtype=float)
        leaves_offsets = np.empty([0], dtype=int)
        leaves_sizes = np.empty([first_dim], dtype=int)
        
        value_position = 0
        
        actual_non_zero_values = 0
        
        for index, leaf_weight in enumerate(weight): # from 0 to first_dim
            
            if (key in ['LEAF_HIDDEN_WEIGHTS', 'LEAF_HIDDEN_BIASES', 'LEAF_OUTPUT_WEIGHTS', 'LEAF_OUTPUT_BIASES']):
                # insert filters non zero values into the fitler sizes
                if (params['FASTINFERENCE'][index] != -1):
                    leaves_sizes[index] = 0
                    continue
            
            # insert filters non zero values into the fitler sizes
            non_zero_values_here = np.count_nonzero(leaf_weight)
            leaves_sizes[index] = non_zero_values_here
            actual_non_zero_values += non_zero_values_here
            # flatten the filter
            flatten_leaf = leaf_weight.ravel()
            offset = 1
            for value in flatten_leaf: # from 0 to (n_depth * n_height * n_width)
                if (value == 0):
                    # increase offset
                    offset += 1
                else:
                    # save value, save index, reset offset, increase position
                    leaves_values = np.append(leaves_values, value)
                    leaves_offsets = np.append(leaves_offsets, offset)
                    leaves_values[value_position] = value
                    leaves_offsets[value_position] = offset
                    offset = 1
                    value_position += 1
            
        tmp = ""
        # substitute the definition
        lines[line] = "#define " + key + "_NNZ " + str(actual_non_zero_values) + "\n"
        line+=1
        lines.insert(
            line,
            "#define " + key + "_DIM " + str(dim) + "\n"
        )
        line+=1
        lines.insert(
            line,
            "#define " + key + "_SPARSE " + str(1) + "\n"
        )
        for d in range(0, dim):
            line+=1
            lines.insert(
                line,
                "#define " + key + "_DIM_" + str(d + 1) + " " + str(weight.shape[d]) + "\n"
            )
        line+=1
        lines.insert(
            line,
            "__hifram fixed " + key + "_data[" + key + "_NNZ] = {\n"
        )
        line+=1
        if flit:
            tmp = ", ".join(["F_LIT(" + str(x) + ")" for x in leaves_values])
        else:
            tmp = ", ".join([str(x) for x in leaves_values])
        lines.insert(
            line,
            tmp
        )
        line+=1
        line+=1
        line+=1
        lines.insert(
            line,
            "\n__hifram fixed " + key + "_offset[" + key + "_NNZ] = {\n"
        )
        line+=1
        if (flit and False):
            tmp = ", ".join(["F_LIT(" + str(x) + ")" for x in leaves_offsets])
        else:
            tmp = ", ".join([str(x) for x in leaves_offsets])
        lines.insert(
            line,
            tmp
        )
        line+=1
        lines.insert(
            line,
            "\n"
        )
        line+=1
        lines.insert(
            line,
            "};\n"
        )
        line+=1
        lines.insert(
            line,
            "\n__hifram fixed " + key + "_sizes[N_LEAVES + 1] = {\n"
        )
        line+=1
        if (flit and False):
            tmp = ", ".join(["F_LIT(" + str(x) + ")" for x in leaves_sizes])
        else:
            tmp = ", ".join([str(x) for x in leaves_sizes])
        lines.insert(
            line,
            tmp
        )
        line+=1
        lines.insert(
            line,
            "\n"
        )
        line+=1
        lines.insert(
            line,
            "};\n\n"
        )

def make_program(run_id, flit=True):
    mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=".")
    wrapped_model = pickle.load(open("./truncated_model.pkl", "rb"))

    node_weights = wrapped_model._fff.fff.node_weights.cpu().detach().numpy()
    node_biases = wrapped_model._fff.fff.node_biases.cpu().detach().numpy()
    w1s = wrapped_model._fff.fff.w1s
    b1s = wrapped_model._fff.fff.b1s.cpu().detach().numpy()
    w2s = wrapped_model._fff.fff.w2s
    b2s = wrapped_model._fff.fff.b2s.cpu().detach().numpy()
    fastinference = wrapped_model._fastinference

    w1s = w1s.transpose(1, 2).cpu().detach().numpy()
    w2s = w2s.transpose(1, 2).cpu().detach().numpy()

    params = {}

    params['NODE_WEIGHTS'] = node_weights
    params['NODE_BIASES'] = node_biases
    params['FASTINFERENCE'] = np.array([-1 if x is None else int(x.argmax()) for x in fastinference])
    actual_leaves_weights = w1s
    actual_leaves_biases = b1s
    actual_leaves_out_weights = w2s
    actual_leaves_out_biases = b2s
    params['LEAF_HIDDEN_WEIGHTS'] = actual_leaves_weights
    params['LEAF_HIDDEN_BIASES'] = actual_leaves_biases
    params['LEAF_OUTPUT_WEIGHTS'] = actual_leaves_out_weights
    params['LEAF_OUTPUT_BIASES'] = actual_leaves_out_biases
    with open("weights.h", "w") as f:
        with open("weights_template.h") as in_f:
            lines = in_f.readlines()

            i = 0
            while i < len(lines):
                i += 1
                if "Add definitions here" in lines[i]:
                    break
            lines[i] = (f"""#define DEPTH {wrapped_model._fff.fff.depth.item()}
#define N_LEAVES {2 ** wrapped_model._fff.fff.depth.item()}
#define INPUT_SIZE {wrapped_model._fff.fff.input_width}
#define LEAF_WIDTH {wrapped_model._fff.fff.leaf_width}
#define OUTPUT_SIZE {wrapped_model._fff.fff.output_width}
#define SIMPLIFIED_LEAVES {sum([f is not None for f in fastinference])}""")
            i+=1
            lines.insert(i, "\n")

            # lines.insert(i+7, """
# float FASTINFERENCE[N_LEAVES] = {-1};
# float NODE_WEIGHTS[N_LEAVES-1][INPUT_SIZE];
# float NODE_BIASES[N_LEAVES-1];
# float LEAF_HIDDEN_WEIGHTS[N_LEAVES-SIMPLIFIED_LEAVES][LEAF_WIDTH][INPUT_SIZE];
# float LEAF_OUTPUT_WEIGHTS[N_LEAVES-SIMPLIFIED_LEAVES][OUTPUT_SIZE][LEAF_WIDTH];
# float LEAF_HIDDEN_BIASES[N_LEAVES-SIMPLIFIED_LEAVES][LEAF_WIDTH];
# float LEAF_OUTPUT_BIASES[N_LEAVES-SIMPLIFIED_LEAVES][OUTPUT_SIZE];
            # """)

            for key in params.keys():
                i = 0
                while i < len(lines):
                    if f"fixed {key}" in lines[i]:
                        break
                    i += 1
                print_parameters(params, key, lines, i)

            f.writelines(lines)
    return wrapped_model


def main(run_id):
    import torch
    net = make_program(run_id)
    net._fff.eval()
    X = np.loadtxt('test.txt')
    X = torch.Tensor(X)
    with open('ref_outputs.txt', 'w') as f:
        y = net(X).argmax(1)
        y = [(str(x) + "\n") for x in y.detach().cpu().numpy()]
        f.writelines(y)


if __name__ == "__main__":
    typer.run(main)
