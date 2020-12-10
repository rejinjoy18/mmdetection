#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=20
NPROC_PER_NODE=8
python -u -m bind_launch --nsockets_per_node=${NSOCKETS_PER_NODE} \
                        --ncores_per_socket=${NCORES_PER_SOCKET} --nproc_per_node=${NPROC_PER_NODE} \
			$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
