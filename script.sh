#!/bin/bash

#SBATCH --job-name=gan_training
#SBATCH --output=gan_training_%j.out
#SBATCH --error=gan_training_%j.err
#SBATCH --partition=savio
#SBATCH --account=loicjannin
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL

srun python -m torch.distributed.launch --nproc_per_node=10 gan.py