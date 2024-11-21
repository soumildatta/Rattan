#!/bin/bash

#SBATCH --time=24:00:00   # Adjust the wall time as necessary
#SBATCH --nodes=1         # Use 1 node
#SBATCH --gres=gpu:h100nvl:1 # Request 1 GPU, adjust depending on availability
#SBATCH --mem=80G         # Mem
#SBATCH --ntasks=1        # Number of tasks, adjust if needed
#SBATCH -o slurm-%j.out-%N   # Output log
#SBATCH -e slurm-%j.err-%N   # Error log
#SBATCH --account=gtao-gpu-np  # Specify your account
#SBATCH --partition=gtao-gpu-np  # Partition to use

export WORKDIR=$SLURM_SUBMIT_DIR

# scratch
export SCRDIR=/scratch/general/vast/$USER/DIAGNOSIS_POKEMON
# rm -rf $SCRDIR
# mkdir -p $SCRDIR

# Copy all the files in the current directory to the scratch directory
# cp -r $WORKDIR/* $SCRDIR
cp -r $WORKDIR/main.py $SCRDIR
# cp -r $WORKDIR/dog_full2 $SCRDIR
cd $SCRDIR

# Load any necessary modules
module use $HOME/MyModules
module load miniforge3/latest

# Activate your virtual environment
source activate diffusion


# training their method
# export MODEL_NAME="stabilityai/stable-diffusion-2" \
# export TRAIN_DATA_DIR="./data/traindata_p0.2_wanet_s1.0_k128_removeeval/train/" \

# for i in {1..10}; do
#   export OUTPUT_DIR="models/output_p0.2_wanet_s1.0_k128_sd2_${i}" \

#   CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$TRAIN_DATA_DIR --caption_column="additional_feature" \
#   --resolution=512 --random_flip \
#   --train_batch_size=24\
#   --num_train_epochs=100 --checkpointing_steps=1000 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=${i} \
#   --output_dir=$OUTPUT_DIR

#   cp -r $OUTPUT_DIR $WORKDIR/models
# done


# for i in {1..10}; do
#   export MODEL_PATH="models/output_p0.2_wanet_s1.0_k128_sd2_${i}"
#   export OUTPUT_DIR="./data/generated_imgs_p0.2_wanet_s1.0_k128_sd2_${i}/"

#   CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path $OUTPUT_DIR --trigger_conditioned

#   cp -r $OUTPUT_DIR $WORKDIR/data
# done


export ORI_DIR="./data/traindata_p0.0_none/train/" \
export COATED_DIR="./data/traindata_p1.0_wanet_unconditional_s1.0_k128/train/" \

# for i in {1..10}; do
#   export GENERATED_INSPECTED_DIR="./data/generated_imgs_p0.2_wanet_s1.0_k128_sd2_${i}/" \

#   CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
#   --coated_dir $COATED_DIR \
#   --generated_inspected_dir $GENERATED_INSPECTED_DIR --trigger_conditioned

#   echo "For POKEMON TRIGGERED malicious on sd2 ($i)"
# done






# Generate the samples with different strength
# CUDA_VISIBLE_DEVICES=0 python main.py --phase generate_img --data_dir data/traindata_p1.0_wanet_unconditional_s2.0_k128/train --save_path data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train_50_sd2 --num_samples 50

# sd2 with our method already generated
# training loop for our method
export MODEL_NAME="stabilityai/stable-diffusion-2"
export TRAIN_DATA_DIR="./data/traindata_p0.2_wanet_s1.0_k128_removeeval/train_50/"

for i in {1..10}; do
  export OUTPUT_DIR="./models/ours_triggered_sd2_${i}"
  export RESUME_DIR="./models/output_p0.2_wanet_s1.0_k128_sd2_${i}/checkpoint-3000"

  CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --caption_column="additional_feature" \
  --resolution=512 --random_flip \
  --train_batch_size=20 \
  --num_train_epochs=30 --checkpointing_steps=2000 \
  --learning_rate=1e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=${i} \
  --resume_from_checkpoint=$RESUME_DIR \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt=None --report_to="wandb"

  cp -r $OUTPUT_DIR $WORKDIR/models
done


for i in {1..10}; do
  export MODEL_PATH="models/ours_triggered_sd2_${i}"
  export OUTPUT_DIR="./data/generated_finetuned_triggered_sd2_${i}/"

  CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path $OUTPUT_DIR --trigger_conditioned

  cp -r $OUTPUT_DIR $WORKDIR/data
done


# # ACTUAL SCRIPT
# # Set environment variables

for i in {1..10}; do
  # Define the GENERATED_INSPECTED_DIR with the incremented number
  GENERATED_INSPECTED_DIR="./data/generated_finetuned_triggered_sd2_${i}/"

  # Run the experiment with the updated GENERATED_INSPECTED_DIR
  CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
    --coated_dir $COATED_DIR \
    --generated_inspected_dir $GENERATED_INSPECTED_DIR --trigger_conditioned

  # Echo message for each iteration
  echo "For POKEMON TRIGGEREDDDD fully OUR METHOD on sd2 ($i)"
done


cd $WORKDIR
# rm -rf $SCRDIR