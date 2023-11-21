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


def make_program(run_id):
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

    params['NODE_WEIGHTS'] = node_weights.flatten()
    params['NODE_BIASES'] = node_biases.flatten()
    params['FASTINFERENCE'] = np.array([-1 if x is None else int(x.argmax()) for x in fastinference])
    actual_leaves_weights = w1s[params['FASTINFERENCE'] == -1]
    actual_leaves_biases = b1s[params['FASTINFERENCE'] == -1]
    actual_leaves_out_weights = w2s[params['FASTINFERENCE'] == -1]
    actual_leaves_out_biases = b2s[params['FASTINFERENCE'] == -1]
    params['LEAF_HIDDEN_WEIGHTS'] = actual_leaves_weights.flatten()
    params['LEAF_HIDDEN_BIASES'] = actual_leaves_biases.flatten()
    params['LEAF_OUTPUT_WEIGHTS'] = actual_leaves_out_weights.flatten()
    params['LEAF_OUTPUT_BIASES'] = actual_leaves_out_biases.flatten()
    # for i, w_arr in enumerate(node_weights):
    #     for j, w in enumerate(w_arr):
    #         params["NODE_WEIGHTS"].append(
    #             f"NODE_WEIGHTS[{i}][{j}] = {w:.8f};\n"
    #         )
    # for i, b in enumerate(node_biases):
    #     params["NODE_BIASES"].append(
    #         f"NODE_BIASES[{i}] = {b[0]:.8f};\n"
    #     )
    # index = 0
    # for i, W in enumerate(w1s):
    #     if fastinference[i] is not None:
    #         params.append(
    #             f"FASTINFERENCE[{i}] = {fastinference[i]};\n"
    #         )
    #     else:
    #         for j, w_arr in enumerate(W):
    #             for k, w in enumerate(w_arr):
    #                 params.append(
    #                     f"LEAF_HIDDEN_WEIGHTS[{index}][{j}][{k}] = {w:.8f};\n"
    #                 )
    #         index += 1
    # index = 0
    # for i, b_arr in enumerate(b1s):
    #     if fastinference[i] is None:
    #         for j, b in enumerate(b_arr):
    #             params.append(
    #                 f"LEAF_HIDDEN_BIASES[{index}][{j}] = {b:.8f};\n"
    #             )
    #         index += 1
    # index = 0
    # for i, W in enumerate(w2s):
    #     if fastinference[i] is None:
    #         for j, w_arr in enumerate(W):
    #             for k, w in enumerate(w_arr):
    #                     params.append(
    #                         f"LEAF_OUTPUT_WEIGHTS[{index}][{j}][{k}] = {w:.8f};\n"
    #                     )
    #         index += 1
    # index = 0
    # for i, b_arr in enumerate(b2s):
    #     if fastinference[i] is None:
    #         for j, b in enumerate(b_arr):
    #             params.append(
    #                 f"LEAF_OUTPUT_BIASES[{index}][{j}] = {b:.8f};\n"
    #             )
    #         index += 1
    with open("weights.h", "w") as f:
        with open("weights_template.h") as in_f:
            lines = in_f.readlines()

            i = 0
            while i < len(lines):
                i += 1
                if "Add definitions here" in lines[i]:
                    break
            lines.insert(i, f"""#define DEPTH {wrapped_model._fff.fff.depth.item()}
#define N_LEAVES {2 ** wrapped_model._fff.fff.depth.item()}
#define INPUT_SIZE {wrapped_model._fff.fff.input_width}
#define LEAF_WIDTH {wrapped_model._fff.fff.leaf_width}
#define OUTPUT_SIZE {wrapped_model._fff.fff.output_width}
#define SIMPLIFIED_LEAVES {sum([f is not None for f in fastinference])}
            """)

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
                i += 1
                lines.insert(
                    i,
                    ", ".join([str(x) for x in params[key]])
                )

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
