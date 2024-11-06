/*
    * weights.h
    *
    *  Created on: Nov 14, 2023
    *      Author: erenyildiz
    */

#ifndef WEIGHTS_H_
#define WEIGHTS_H_
#include "fixed.h"
#include "mem.h"

// Add definitions here

__hifram fixed FASTINFERENCE[N_LEAVES] = {

};

__hifram fixed NODE_WEIGHTS[N_LEAVES - 1][INPUT_SIZE] = {

};
__hifram fixed NODE_BIASES[N_LEAVES - 1] = {

};

__hifram fixed LEAF_HIDDEN_WEIGHTS[N_LEAVES][LEAF_WIDTH][INPUT_SIZE] = {

};

__hifram fixed LEAF_HIDDEN_BIASES[N_LEAVES][LEAF_WIDTH] = {

};

__hifram fixed LEAF_OUTPUT_WEIGHTS[N_LEAVES][OUTPUT_SIZE][LEAF_WIDTH] = {

};

__hifram fixed LEAF_OUTPUT_BIASES[N_LEAVES][OUTPUT_SIZE] = {

};

#endif /* WEIGHTS_H_ */

