#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from .sampler_invsr import InvSamplerSR, BaseSampler

from .utils import util_common
from .utils.util_opts import str2bool
from huggingface_hub import hf_hub_download
from shutil import copy2

class Namespace:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        items = [f"{key}={repr(value)}" for key, value in vars(self).items()]
        return f"Namespace({', '.join(items)})"

def get_configs(args, log=False):
    configs = OmegaConf.load(args.cfg_path)

    if args.timesteps is not None:
        assert len(args.timesteps) == args.num_steps
        configs.timesteps = sorted(args.timesteps, reverse=True)
    else:
        if args.num_steps == 1:
            configs.timesteps = [200,]
        elif args.num_steps == 2:
            configs.timesteps = [200, 100]
        elif args.num_steps == 3:
            configs.timesteps = [200, 100, 50]
        elif args.num_steps == 4:
            configs.timesteps = [200, 150, 100, 50]
        elif args.num_steps == 5:
            configs.timesteps = [250, 200, 150, 100, 50]
        else:
            assert args.num_steps <= 250
            configs.timesteps = np.linspace(
                start=args.started_step, stop=0, num=args.num_steps, endpoint=False, dtype=np.int64()
            ).tolist()
    if log:
        print(f'[InvSR] - Setting timesteps for inference: {configs.timesteps}')

    # path to save Stable Diffusion
    sd_path = args.sd_path if args.sd_path else "./weights"
    util_common.mkdir(sd_path, delete=False, parents=True)
    configs.sd_pipe.params.cache_dir = sd_path

    # path to save noise predictor
    started_ckpt_name = args.invsr_model

    if getattr(args, "started_ckpt_dir", None) is not None:
        started_ckpt_dir = args.started_ckpt_dir
    else:
        started_ckpt_dir = "./weights"

    if getattr(args, "started_ckpt_path", None) is not None:
        started_ckpt_path = args.started_ckpt_path
    else:
        started_ckpt_path = Path(started_ckpt_dir) / started_ckpt_name
        util_common.mkdir(started_ckpt_dir, delete=False, parents=True)

    if not Path(started_ckpt_path).exists():
        temp_path = hf_hub_download(
            repo_id="OAOA/InvSR",
            filename=started_ckpt_name,
        )
        copy2(temp_path, started_ckpt_path)
    configs.model_start.ckpt_path = str(started_ckpt_path)

    configs.bs = args.bs
    configs.tiled_vae = args.tiled_vae
    configs.color_fix = args.color_fix
    configs.basesr.chopping.pch_size = args.chopping_size
    if args.bs > 1:
        configs.basesr.chopping.extra_bs = 1
    else:
        configs.basesr.chopping.extra_bs = args.chopping_bs

    return configs
