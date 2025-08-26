#!/bin/bash

#SBATCH --time=23:59:00
#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)


# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
# module load pytorch/2.5.1-py312-cu124-mpi-sota
# module load pytorch/2.1.0-py312-cu122-mpi
module load tensorflow/2.16.1-pip-py312-cuda122
module load parallel python
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate


CUDA_VISIBLE_DEVICES=0 python3 test_temporal_integration.py
