#!/bin/bash
# This script produces Navier-Stokes trajectories using SLURM job arrays for parallel processing.

#SBATCH --time=01:00:00
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH --account=OD-230881
#SBATCH --cpus-per-task=8
#SBATCH --array=0-7              # Run 8 jobs in parallel (0, 1, 2, 3, 4, 5, 6, 7)


module load pytorch/2.5.1-py312-cu122-mpi
source $HOME/.venvs/pytorch/bin/activate

# Get the SLURM array task ID (0-7)
TASK_ID=$SLURM_ARRAY_TASK_ID

# Total samples and samples per job
TOTAL_SAMPLES=8  # Adjust this to your total number of samples
SAMPLES_PER_JOB=$((TOTAL_SAMPLES / 8))  # 8 parallel jobs

# Calculate the range for this job
START_SAMPLE=$((TASK_ID * SAMPLES_PER_JOB))
END_SAMPLE=$((START_SAMPLE + SAMPLES_PER_JOB - 1))

# For the last job, handle any remaining samples
if [ $TASK_ID -eq 7 ]; then
    END_SAMPLE=$((TOTAL_SAMPLES - 1))
fi

echo "Job $TASK_ID: Processing samples $START_SAMPLE to $END_SAMPLE"

# Run the data generation with sample range
python3 generate_data.py \
    base=configs/navierstokes2dsmoke_nt2800_tol3e-4.yaml \
    experiment=smoke \
    mode=train \
    samples=$SAMPLES_PER_JOB \
    seed=$((197910 + TASK_ID)) \
    dirname=/scratch3/wan410/operator_learning_data/pdearena/NSE-2D-Customised \
    start_sample=$START_SAMPLE

echo "Job $TASK_ID completed: samples $START_SAMPLE to $END_SAMPLE"