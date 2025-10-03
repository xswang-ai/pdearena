#!/bin/bash
# This script produces 5.2k training, 1.3k valid, and 1.3k test trajectories of the Navier-Stokes dataset.

#SBATCH --time=00:20:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)


# module load numpy/2.0.0-py312
module load pytorch/2.5.1-py312-cu122-mpi
module load parallel python
# source /scratch3/wan410/venvs/testing/bin/activate
source $HOME/.venvs/pytorch/bin/activate


python3 generate_data.py base=configs/navierstokes2dsmoke_nt5k_fy0.1.yaml  experiment=smoke mode=train samples=5000 seed=197910 \
dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised

# python3 generate_data.py base=configs/navierstokes2dsmoke_nt5k_fy0.1.yaml \
# experiment=smoke mode=train samples=5000 seed=197910 pdeconfig.init_args.device=cpu\
# dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised