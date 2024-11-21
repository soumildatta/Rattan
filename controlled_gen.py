from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from torchvision.utils import save_image

import argparse
import json
import numpy as np
import os
import shutil
import torch
import torchvision.transforms as T
import torch.nn.functional as F


def mean_conv2d(input_tensor, kernel_size, stride=1, padding=0):
    in_channels, height, width = input_tensor.size()
    mean_kernel = torch.ones(in_channels, 1, kernel_size, kernel_size)\
                        / (kernel_size * kernel_size)

    return F.conv2d(input_tensor, mean_kernel, stride=stride, padding=padding,
                    groups=in_channels)


def add_gaussian_noise(image_tensor, mean=0.0, std=0.1):
    noise = torch.randn(image_tensor.size()) * std + mean
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image


def transform(src, dst):
    img = T.ToTensor()(Image.open(src))

    # 1. add gaussion noise
    # img = add_gaussian_noise(img, std=0.3)

    # 2. gaussian blur
    # blurrer = T.GaussianBlur(kernel_size=(127, 127), sigma=(0.1, 5.0))
    # img = blurrer(img)

    # 3. mean filter
    img = T.Resize(size=img.size(1)*3)(img)
    img = mean_conv2d(img, 9, stride=3)
    img = mean_conv2d(img, 7, stride=2)
    img = mean_conv2d(img, 5, stride=2)
    img = mean_conv2d(img, 3, stride=1)
    sharpener = T.RandomAdjustSharpness(sharpness_factor=5)
    img = sharpener(img)

    save_image(img, dst)


def process_data(args):
    base_dir = f'{args.data_dir}/train'
    path = f'{args.data_dir}/train_{args.num_samples}' 
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{path}/metadata.jsonl', 'w') as f:
        i = 0
        for line in open(f'{base_dir}/metadata.jsonl'):
            if i > args.num_samples - 1:
                break

            f.write(line)
            if args.transform:
                transform(f'{base_dir}/{i}.png', f'{path}/{i}.png')
            else:
                shutil.copyfile(f'{base_dir}/{i}.png', f'{path}/{i}.png')

            i += 1


def generate_img(args):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    # pipe.unet.load_attn_procs(args.model_path)
    pipe.to('cuda')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    names = []
    prompts = []
    with open(f'{args.data_dir}/metadata.jsonl', 'r') as f:
        with open(os.path.join(args.save_path, 'metadata.jsonl'), 'w') as f_out:
            i = 0
            for line in f:
                json_obj = json.loads(line)
                names.append(json_obj['file_name'])
                prompts.append(json_obj['additional_feature'])

                if i < args.num_samples:
                    f_out.write(line)

                i += 1

    bs = 1
    for i in range(int(np.ceil(len(prompts)/bs))):
        texts = prompts[i*bs : (i+1)*bs]
        init_imgs = []
        sizes = []
        for j in range(bs):
            img = T.ToTensor()(Image.open(f'{args.data_dir}/{names[i*bs+j]}'))
            sizes.append(img.size(1))
            # img = T.Resize(size=img.size(1)*5)(img)
            # img = mean_conv2d(img, 9, stride=3)
            # img = mean_conv2d(img, 7, stride=2)
            # img = mean_conv2d(img, 5, stride=2)
            # img = mean_conv2d(img, 3, stride=1)
            img = T.Resize(size=1280)(img)
            init_imgs.append(img)

        # recover watermark
        # imgs = pipe(texts, init_imgs, strength=0.6, num_inference_steps=10,
        #             guidance_scale=7.5).images

        imgs = pipe(texts, init_imgs, strength=args.strength, num_inference_steps=100,
                    guidance_scale=1).images

        for j in range(bs):
            # imgs[j] = T.Resize(size=sizes[j])(imgs[j])
            imgs[j].save(os.path.join(args.save_path, names[i*bs+j]))

        print((i+1)*bs)

        if (i+1)*bs >= args.num_samples:
            exit()


def main(args):
    if args.phase == 'process_data':
        process_data(args)
    elif args.phase == 'generate_img':
        generate_img(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--transform', action='store_true')
    parser.add_argument('--strength', type=float, default=0.6)

    args = parser.parse_args()

    main(args)
