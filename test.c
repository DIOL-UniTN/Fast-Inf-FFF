#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
#define DEPTH 4
#define N_LEAVES 16
#define INPUT_SIZE 784
#define LEAF_WIDTH 32
#define OUTPUT_SIZE 10


float NODE_WEIGHTS[N_LEAVES-1][INPUT_SIZE];
float NODE_BIASES[N_LEAVES-1];
float LEAF_HIDDEN_WEIGHTS[N_LEAVES][LEAF_WIDTH][INPUT_SIZE];
float LEAF_OUTPUT_WEIGHTS[N_LEAVES][OUTPUT_SIZE][LEAF_WIDTH];
float LEAF_HIDDEN_BIASES[N_LEAVES][LEAF_WIDTH];
float LEAF_OUTPUT_BIASES[N_LEAVES][LEAF_WIDTH];
*/

float neuron(float* weights, float bias, float* inputs, int input_size) {
    /**
     * Compute the output of a neuron using ReLU as activation.
     * :weights: the weights of the neuron.
     * :bias: the bias of the neuron.
     * :inputs: the inputs to the neuron.
     */
    float accumulator = 0;
    for (int i = 0; i < input_size; ++i) {
        accumulator += weights[i] * inputs[i];
    }
    accumulator += bias;
    return accumulator;
}

int leaf(int leaf_id, float* input) {
    /*
     * Compute the output of a leaf
     * :leaf_id: the id (index) of the leaf
     * :input: the input to the leaf
     */
    float hidden[LEAF_WIDTH];
    float output[OUTPUT_SIZE];
    int argmax = 0;
    if (FASTINFERENCE[leaf_id] != -1) {
        return FASTINFERENCE[leaf_id];
    } else {
        for (int i = 0; i < N_LEAVES; i++) {
            if (FASTINFERENCE[i] != -1) {
                leaf_id++;
            }
        }
        for (int i = 0; i < LEAF_WIDTH; ++i) {
            hidden[i] = neuron(
                LEAF_HIDDEN_WEIGHTS[leaf_id][i],
                LEAF_HIDDEN_BIASES[leaf_id][i],
                input,
                INPUT_SIZE
            );
            hidden[i] = hidden[i] < 0 ? 0 : hidden[i];
        }
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            output[i] = neuron(
                LEAF_OUTPUT_WEIGHTS[leaf_id][i],
                LEAF_OUTPUT_BIASES[leaf_id][i],
                hidden,
                LEAF_WIDTH
            );
        }
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            if (output[i] > output[argmax]) {
                argmax = i;
            }
        }
        return argmax;
    }
}

int tree_inference(float* input) {
    int cur_node = 0;
    float output = 0.0;
    int platform = 0;
    int next_platform = 0;
    int choice = 0;
    
    for (int depth = 0; depth < DEPTH; depth++) {
        output = neuron(
            NODE_WEIGHTS[cur_node],
            NODE_BIASES[cur_node],
            input,
            INPUT_SIZE
        );

        platform = pow(2,depth) - 1;
        next_platform = pow(2, (depth + 1)) - 1;
        choice = output >= 0 ? 1 : 0;
        cur_node = (cur_node - platform) * 2 + choice + next_platform;
    }
    cur_node = (cur_node - next_platform);
    return leaf(cur_node, input);
}

int main(int argc, char* argv[]) {
    // Insert weights here

    float x[784] = {0};
    for (int i = 1; i < argc; i++) {
        x[i-1] = atof(argv[i]);
    }
    printf("%d ", tree_inference(x));
}
