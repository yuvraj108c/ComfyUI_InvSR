from .InvSR.inference_invsr import get_configs
from .InvSR.sampler_invsr import InvSamplerSR, BaseSampler
import torch
from comfy.utils import ProgressBar

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

class Namespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        items = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return f"Namespace({', '.join(items)})"

class LoadInvSRModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sd_model": (['stabilityai/sd-turbo'],),
                "invsr_model": (['noise_predictor_sd_turbo_v5.pth'],),
                "tiled_vae": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("INVSR_PIPE",)
    RETURN_NAMES = ("invsr_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "INVSR"

    def loadmodel(self, sd_model, invsr_model, tiled_vae):
        
        args = Namespace(bs=1, chopping_bs=8, timesteps=None, num_steps=1, cfg_path='custom_nodes/ComfyUI_InvSR/InvSR/configs/sample-sd-turbo.yaml', sd_path='models/diffusers', started_ckpt_path='custom_nodes/ComfyUI_InvSR/weights/noise_predictor_sd_turbo_v5.pth', tiled_vae=tiled_vae, color_fix='', chopping_size=128)
        configs = get_configs(args)
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

    def process(self, invsr_pipe, images, num_steps, batch_size, chopping_batch_size, chopping_size, color_fix, seed):
        base_sampler = invsr_pipe
        if color_fix == "none":
            color_fix = ""

        args = Namespace(bs=batch_size, chopping_bs=chopping_batch_size, timesteps=None, num_steps=num_steps, cfg_path='custom_nodes/ComfyUI_InvSR/InvSR/configs/sample-sd-turbo.yaml', sd_path='models/diffusers', started_ckpt_path='custom_nodes/ComfyUI_InvSR/weights/noise_predictor_sd_turbo_v5.pth', tiled_vae=base_sampler.configs.tiled_vae, color_fix=color_fix, chopping_size=chopping_size)
        configs = get_configs(args)
        base_sampler.configs = get_configs(args, log=True)
        base_sampler.setup_seed(seed)
        sampler = InvSamplerSR(base_sampler)

        images_bchw = images.permute(0,3,1,2)
        batches = split_tensor_into_batches(images_bchw, batch_size)

        results = []
        pbar = ProgressBar(len(batches))

        for batch in batches:
            result = sampler.inference(image_bchw=batch)
            results.append(torch.from_numpy(result))
            pbar.update(1)

        return (torch.cat(results, dim=0),)