#!/usr/bin/env sh

conda activate fl;

python3 fedavg_server.py &\ ;

for i in {0..9}; do python3 normal_ff_client.py $i &\ ; done;

wait
