from .comfyui_invsr_trimmed import get_configs, InvSamplerSR, BaseSampler, Namespace
import torch
from comfy.utils import ProgressBar
from folder_paths import get_full_path, get_folder_paths, models_dir
import os
import torch.nn.functional as F

def split_tensor_into_batches(tensor, batch_size):
    """
    Split a tensor into smaller batches of specified size
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (N, C, H, W)
        batch_size (int): Desired batch size for splitting
        
    Returns:
        list: List of tensors, each with batch_size (except possibly the last one)
    """
    # Get original batch size
    original_batch_size = tensor.size(0)
    
    # Calculate number of full batches and remaining samples
    num_full_batches = original_batch_size // batch_size
    remaining_samples = original_batch_size % batch_size
    
    # Split tensor into chunks
    batches = []
    
    # Handle full batches
    for i in range(num_full_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = tensor[start_idx:end_idx]
        batches.append(batch)
    
    # Handle remaining samples if any
    if remaining_samples > 0:
        last_batch = tensor[-remaining_samples:]
        batches.append(last_batch)
    
    return batches


class LoadInvSRModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd_model": (['stabilityai/sd-turbo'],),
                "invsr_model": (['noise_predictor_sd_turbo_v5.pth'],),
                "dtype": (['fp16', 'fp32', 'bf16'], {"default": "fp16"}),
                "tiled_vae": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("INVSR_PIPE",)
    RETURN_NAMES = ("invsr_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "INVSR"

    def loadmodel(self, sd_model, invsr_model, dtype, tiled_vae):
        match dtype:
            case "fp16":
                dtype = "torch.float16"
            case "fp32":
                dtype = "torch.float32"
            case "bf16":
                dtype = "torch.bfloat16"

        cfg_path = os.path.join(
            os.path.dirname(__file__), "configs", "sample-sd-turbo.yaml"
        )
        sd_path = get_folder_paths("diffusers")[0]

        try:
            ckpt_dir = get_folder_paths("invsr")[0]
        except:
            ckpt_dir = os.path.join(models_dir, "invsr")

        args = Namespace(
            bs=1,
            chopping_bs=8,
            timesteps=None,
            num_steps=1,
            cfg_path=cfg_path,
            sd_path=sd_path,
            started_ckpt_dir=ckpt_dir,
            tiled_vae=tiled_vae,
            color_fix="",
            chopping_size=128,
        )
        configs = get_configs(args)
        configs["sd_pipe"]["params"]["torch_dtype"] = dtype
        base_sampler = BaseSampler(configs)

        return (base_sampler,)

class InvSRSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "invsr_pipe": ("INVSR_PIPE",),
                "images": ("IMAGE",),
                "num_steps": ("INT",{"default": 1, "min": 1, "max": 5}),
                "cfg": ("FLOAT",{"default": 1.0, "step":0.1}),
                # "scale_factor": ("INT",{"default": 4}),
                "batch_size": ("INT",{"default": 1}),
                "chopping_batch_size": ("INT",{"default": 8}),
                "chopping_size": ([128, 256, 512],{"default": 128}),
                "color_fix": (['none', 'wavelet', 'ycbcr'], {"default": "none"}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 2**32 - 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "INVSR"

    def process(self, invsr_pipe, images, num_steps, cfg, batch_size, chopping_batch_size, chopping_size, color_fix, seed):
        base_sampler = invsr_pipe
        if color_fix == "none":
            color_fix = ""

        cfg_path = os.path.join(
            os.path.dirname(__file__), "configs", "sample-sd-turbo.yaml"
        )
        sd_path = get_folder_paths("diffusers")[0]

        try:
            ckpt_dir = get_folder_paths("invsr")[0]
        except:
            ckpt_dir = os.path.join(models_dir, "invsr")

        args = Namespace(
            bs=batch_size,
            chopping_bs=chopping_batch_size,
            timesteps=None,
            num_steps=num_steps,
            cfg_path=cfg_path,
            sd_path=sd_path,
            started_ckpt_dir=ckpt_dir,
            tiled_vae=base_sampler.configs.tiled_vae,
            color_fix=color_fix,
            chopping_size=chopping_size,
        )
        configs = get_configs(args, log=True)
        configs["cfg_scale"] = cfg
        # configs["basesr"]["sf"] = scale_factor
        
        base_sampler.configs = configs
        base_sampler.setup_seed(seed)
        sampler = InvSamplerSR(base_sampler)

        images_bchw = images.permute(0,3,1,2)
        og_h, og_w = images_bchw.shape[2:]

        # Calculate new dimensions divisible by 16
        new_height = ((og_h + 15) // 16) * 16  # Round up to nearest multiple of 16
        new_width = ((og_w + 15) // 16) * 16
        resized = False
        
        if og_h != new_height or og_w != new_width:
            resized = True
            print(f"[InvSR] - Image not divisible by 16. Resizing to {new_height} (h) x {new_width} (w)")
            images_bchw = F.interpolate(images_bchw, size=(new_height, new_width), mode='bicubic', align_corners=False)

        batches = split_tensor_into_batches(images_bchw, batch_size)

        results = []
        pbar = ProgressBar(len(batches))

        for batch in batches:
            result = sampler.inference(image_bchw=batch)
            results.append(torch.from_numpy(result))
            pbar.update(1)

        result_t = torch.cat(results, dim=0)

        # Resize to original dimensions * 4
        if resized:
            result_t = F.interpolate(result_t, size=(og_h * 4, og_w * 4), mode='bicubic', align_corners=False)
            
        return (result_t.permute(0,2,3,1),)
