import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import glob
from pathlib import Path
from PIL import Image
import numpy as np

from Flow.flow_comp_raft import RAFT_bi


@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "sample"), exist_ok=True)


    config = OmegaConf.load(args.config)

    # Create validation pipeline
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()
    Flow_estimator = RAFT_bi().cuda()

    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)
        # Add the new parameters with fallback to args values
        model_config.frames_path = model_config.get("frames_path", args.frames_path)
        model_config.masks_path = model_config.get("masks_path", args.masks_path)
        model_config.first_frame_path = model_config.get("first_frame_path", args.first_frame_path)

        inference_config = OmegaConf.load(model_config.get("inference_config"))
        pretrained_model_path = r"D:\.cache\hub\models--stable-diffusion-v1-5--stable-diffusion-inpainting\snapshots\8a4288a76071f7280aedbdb3253bdb9e9d5d84bb"
        unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet",
                                                       unet_additional_kwargs=OmegaConf.to_container(
                                                           inference_config.unet_additional_kwargs)).cuda()

        # Load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""

            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get(
                "controlnet_additional_kwargs", {}))

            print(f"Loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict[
                "controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): 
                image_paths = [image_paths]

            print(f"Controlnet image paths:")
            for path in image_paths: 
                print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0),
                    ratio=(model_config.W / model_config.H, model_config.W / model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else:
                image_norm = lambda x: x

            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1, 2, 0))).astype(np.uint8)).save(
                    f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # Set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            if controlnet is not None: 
                controlnet.enable_xformers_memory_efficient_attention()
        pipeline = AnimationPipeline(
            vae=vae.half(), 
            text_encoder=text_encoder.half(), 
            tokenizer=tokenizer, 
            unet=unet.half(),
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
            Flow_estimator=Flow_estimator.half(),
        )
        pipeline = pipeline.to("cuda")
        pipeline = load_weights(
            pipeline,
            # Motion module
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            # Domain adapter
            adapter_lora_path=model_config.get("adapter_lora_path", ""),
            adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
            # Image layers
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to("cuda")
        pipeline.set_use_memory_efficient_attention_xformers(True)
        pipeline.enable_xformers_memory_efficient_attention()

        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(
            model_config.n_prompt) == 1 else model_config.n_prompt

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

        config[model_idx].random_seed = []
        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
            # Manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())

            print(f"Current seed: {torch.initial_seed()}")
            print(f"Sampling '{prompt}' ...")
            
            # Define output video path based on prompt
            safe_prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            
            sample = pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,
                
                # Use parameters from model_config
                frames_path=model_config.frames_path,
                masks_path=model_config.masks_path,
                first_frame_path=model_config.first_frame_path,
                
                # ControlNet parameters
                controlnet_images=controlnet_images,
                controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
            ).videos

            # Save the output as GIF as well
            gif_path = f"{savedir}/sample/{sample_idx}-{safe_prompt}.gif"
            save_videos_grid(sample, gif_path)
            print(f"Saved to {gif_path} ")

            sample_idx += 1


    # Save configuration
    OmegaConf.save(config, f"{savedir}/config.yaml")
    print(f"All outputs saved to {savedir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--pretrained-model-path", type=str,
                         default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--without-xformers", action="store_true")
    
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=720)
    parser.add_argument("--H", type=int, default=360)
    
    
    parser.add_argument("--frames-path", type=str,  
                        help="Path to the directory containing video frames")
    parser.add_argument("--masks-path", type=str, 
                        help="Path to the directory containing mask frames")
    parser.add_argument("--first-frame-path", type=str, 
                        help="Path to the first frame image file")

    args = parser.parse_args()
    main(args)