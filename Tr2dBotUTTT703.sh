#!/bin/bash

#SBATCH --job-name=nnunet-703-eval
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
module load anaconda/2023.03

#eval "$(/lustre/apps/apps/anaconda3/anaconda3-2023.03/bin/conda shell.bash hook)"
#conda activate llm360
# ldconfig /.singularity.d/lib
source /lustre/scratch/users/guowei.he/scripts/load-apptainer.sh
source /lustre/apps/apps/anaconda3/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate uTTT

# nnUNetv2_plan_and_preprocess -d 703 --verify_dataset_integrity

# nnUNetv2_train 703 2d all -tr nnUNetTrainerUMambaBot

# nnUNetv2_predict -i /lustre/scratch/users/hongyi.wang/U-Mamba/data/nnUNet_raw/Dataset703_NeurIPSCell/imagesVal -o evaluation_results -d 703 -c 2d -f all -tr nnUNetTrainerUMambaBot --disable_tta

/home/hongyi.wang/.conda/envs/uTTT/bin/python evaluation/compute_cell_metric.py --gt_path data/nnUNet_raw/Dataset703_NeurIPSCell/labelsVal-instance-mask --seg_path evaluation_results
