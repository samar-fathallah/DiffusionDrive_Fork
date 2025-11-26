#!/bin/bash -l
#SBATCH --job-name=diffusion_drive_infer
#SBATCH --output=diffusion_drive_infer_%j.out
#SBATCH --error=diffusion_drive_infer_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=60:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --qos=batch

module load cuda/11.2
pyenv activate thesis_venv

cd /no_backups/s1479/DiffusionDrive

export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python /no_backups/s1479/DiffusionDrive/inference/test_runner.py \
    --config /no_backups/s1479/DiffusionDrive/projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
    --checkpoint /no_backups/s1479/DiffusionDrive/ckpts/diffusiondrive_nusc_stage2.pth \
    --ann_file /no_backups/s1479/DiffusionDrive/data/infos/nuscenes_infos_val.pkl \
    --data_root data/nuscenes \
    --frame_idx 600

# python /no_backups/s1479/DiffusionDrive/inference/steering.py data/infos/nuscenes_infos_val.pkl --limit 2000