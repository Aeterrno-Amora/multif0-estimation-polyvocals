#!/bin/bash
#
#SBATCH --job-name=augmentation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=augm.out
#SBATCH --error=augm.err


module purge

module load cudnn/8.1
module load cuda/10.2.89

module load rubberband/intel/1.8.1
module load ffmpeg/intel/3.2.2

#python data_augmentation.py --f0-path /scratch/hc2945/data/BachChorales/BC/pyin_annot --audio-path /scratch/hc2945/data/BachChorales/BC --dataset BC
#echo BC done!

#python data_augmentation.py --f0-path /scratch/hc2945/data/BarbershopQuartets/BQ/pyin_annot --audio-path /scratch/hc2945/data/BarbershopQuartets/BQ --dataset BSQ
#echo BSQ done!

python experiments/data_augmentation.py --f0-path /root/multif0-estimation-polyvocals/data/ChoralSingingDataset/ --audio-path /root/multif0-estimation-polyvocals/data/ChoralSingingDataset/ --dataset CSD
echo CSD done!

python experiments/data_augmentation.py --f0-path /root/multif0-estimation-polyvocals/data/EsmucChoirDataset_v1.0.0/ --audio-path /root/multif0-estimation-polyvocals/data/EsmucChoirDataset_v1.0.0/ --dataset ECS
echo ECS done!

python experiments/data_augmentation.py --f0-path /root/multif0-estimation-polyvocals/data/DagstuhlChoirSet_V1.2.3/annotations_csv_F0_PYIN --audio-path /root/multif0-estimation-polyvocals/data/DagstuhlChoirSet_V1.2.3/audio_wav_22050_mono --dataset DCS
echo DCS done!