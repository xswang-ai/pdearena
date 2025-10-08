#!/bin/bash
# This script produces Navier-Stokes trajectories using SLURM job arrays for parallel processing.

#SBATCH --time=00:10:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)


module load pytorch/2.5.1-py312-cu122-mpi
source $HOME/.venvs/pytorch/bin/activate

# python3 generate_data.py base=configs/navierstokes2dsmoke_nt560_tol1e-3.yaml  experiment=smoke mode=train samples=2  seed=197910 \
# dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised

python3 test_training_mcwilliams.py