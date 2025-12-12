#!/bin/bash

#SBATCH --job-name=nnunet-704-2d-train
#SBATCH --partition=gpumid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --reservation=training

module load cuda/12.1
#eval "$(/lustre/apps/apps/anaconda3/anaconda3-2023.03/bin/conda shell.bash hook)"
#conda activate llm360
# ldconfig /.singularity.d/lib
source /lustre/scratch/users/guowei.he/scripts/load-apptainer.sh

conda activate uTTT

# nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity

nnUNetv2_train 704 2d all -tr nnUNetTrainerUMambaEnc
