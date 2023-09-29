#!/usr/bin/bash

conda activate fl;

python3 fedavg_server.py &\ ;

for i in {0..9}; do echo $i; python3 fed_client.py $i $1 $2 &\ ; done;

wait
