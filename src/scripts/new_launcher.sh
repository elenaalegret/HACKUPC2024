#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH -D .
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:2
#SBATCH --qos=acc_debug

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
module load spack

python main.py
