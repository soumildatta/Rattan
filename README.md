# Rattan

Official repository for "Towards Evaluating the Robustness of Watermark-Based Protections against Unauthorized Data Usage in Text-to-Image Diffusion Models"

## Datasets 
The following datasets were used for evaluating RATTAN:       
[Naruto Dataset](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions)\
[Pokemon Dataset](https://huggingface.co/datasets/reach-vb/pokemon-blip-captions)\
[CelebA Dataset](https://huggingface.co/datasets/irodkin/celeba_with_llava_captions)

## Environment
To get started quickly, utilize the __requirements.txt__ file included in the repository. The experimentations were performed with Python v3.10.15. 
Create a virtual environment, then install the dependencies using:
```sh
pip install -r requirements.txt
```

## Usage 

### Generating a watermarked model using DIAGNOSIS
To test the efficacy of RATTAN, first we need to train a watermarked model. The code for the DIAGNOSIS framework is included for convenience with minor changes for ease of use. The code was adapted from the [official repository of DIAGNOSIS](https://github.com/ZhentingWang/DIAGNOSIS).

To train a watermarked model, first enter the DIAGNOSIS folder, and run the commands from the instructions in the [official repository](https://github.com/ZhentingWang/DIAGNOSIS).

One difference to the original repository is to train the binary classifier before using it.
To that regard, train the binary classifier in the same way you would use binary_classifier.py:
```sh
export ORI_DIR="./traindata_p0.0_none/train/"
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128/train/"
export GENERATED_INSPECTED_DIR="./generated_imgs_p1.0_wanet_unconditional_s2.0_k128/"

CUDA_VISIBLE_DEVICES=0 python train_binary_classifier.py --ori_dir $ORI_DIR --coated_dir $COATED_DIR --generated_inspected_dir $GENERATED_INSPECTED_DIR
```

This makes it easier, and you can use these trained classifiers in the future without having to train from scratch every time.

### Using RATTAN
Once you have the watermark-memorized model, we can then use RATTAN to remove the injected memorization from the model through the following steps:

1. In the home directory of the repository, run controlled_gen.py on the watermarked data to perform controlled generation of the watermarked data:
```sh
CUDA_VISIBLE_DEVICES=0 python controlled_gen.py --phase generate_img --data_dir <coated_dir/train> --save_path <coated_dir/train_10> --num_samples 10 --model_path CompVis/stable-diffusion-v1-4
```

Replace 'coated_dir' with the DIAGNOSIS generated coated images without the evaluation set. For example, for unconditional injected memorization, the 'coated_dir' would be __traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval__.

Next, finetune the injected model with this new controlled generated data:

``` sh
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DATA_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train_50/"
export OUTPUT_DIR="./stable_50_lr1e-5_p1.0_wanet_unconditional_s2.0_k128"
export RESUME_DIR="./output_p1.0_wanet_unconditional_s2.0_k128/checkpoint-75000"

CUDA_VISIBLE_DEVICES=1 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DATA_DIR \
--caption_column="additional_feature" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=30 --checkpointing_steps=2000 \
--learning_rate=1e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--resume_from_checkpoint=$RESUME_DIR \
--output_dir=$OUTPUT_DIR \
--validation_prompt=None --report_to="wandb"
```

Once trained, the RATTAN generated cleaned model is saved in __stable_50_lr1e-5_p1.0_wanet_unconditional_s2.0_k128__. The results of the model can then be checked in a manner similar to DIAGNOSIS -- by generating samples, and utilizing the binary classifier.

To generate images from the RATTAN finetuned model:
```sh
export MODEL_PATH="stable_50_lr1e-5_p1.0_wanet_unconditional_s2.0_k128"
export OUTPUT_DIR="generated_finetuned_imgs/"

CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path $OUTPUT_DIR
```

Then, use the binary classifier that was trained during DIAGNOSIS (you do not need to retrain the binary classifier):

```sh
export ORI_DIR="./traindata_p0.0_none/train/"
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128/train/"
export GENERATED_INSPECTED_DIR="./generated_finetuned_imgs/"

CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR --coated_dir $COATED_DIR --generated_inspected_dir $GENERATED_INSPECTED_DIR
```

This will output the FID score, the memorization strength, and finally whether the model is benign or not. 

## Additional Scripts
Additional scripts to perform simple image transformations on the dataset are available under the _Transformations_ folder.

## Acknowledgements 
Part of this codebase was adapted from the official DIAGNOSIS repository: https://github.com/ZhentingWang/DIAGNOSIS, and the training script was sourced from the Huggingface Diffusers library examples at https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py. 