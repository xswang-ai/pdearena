#!/bin/bash
# This script produces 5.2k training, 1.3k valid, and 1.3k test trajectories of the Navier-Stokes dataset.

#SBATCH --time=00:30:00

#SBATCH --mem=256gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)


module load pytorch/2.5.1-py312-cu122-mpi
source $HOME/.venvs/pytorch/bin/activate

# python3 generate_data.py base=configs/navierstokes2dsmoke_nt560_tol1e-3.yaml  experiment=smoke mode=train samples=2  seed=197910 \
# dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised

python3 generate_data.py base=configs/navierstokes2dsmoke_nt4200_tol3e-4.yaml  experiment=smoke mode=train samples=2  seed=197910 \
dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised