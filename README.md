<div align="center">

# ComfyUI InvSR
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.09013-b31b1b.svg)](https://arxiv.org/abs/2412.09013) 

This project is a ComfyUI wrapper for [InvSR](https://github.com/zsyOAOA/InvSR) (Arbitrary-steps Image Super-resolution via Diffusion Inversion)

<img height="400" src="https://github.com/user-attachments/assets/6c057a3c-3355-4060-9161-a88ab6f6d986" />

</div>

## ⭐ Support
If you like my projects and wish to see updates and new features, please consider supporting me. It helps a lot! 

[![ComfyUI-Depth-Anything-Tensorrt](https://img.shields.io/badge/ComfyUI--Depth--Anything--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Depth-Anything-Tensorrt)
[![ComfyUI-Upscaler-Tensorrt](https://img.shields.io/badge/ComfyUI--Upscaler--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Upscaler-Tensorrt)
[![ComfyUI-Dwpose-Tensorrt](https://img.shields.io/badge/ComfyUI--Dwpose--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Dwpose-Tensorrt)
[![ComfyUI-Rife-Tensorrt](https://img.shields.io/badge/ComfyUI--Rife--Tensorrt-blue?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt)

[![ComfyUI-Whisper](https://img.shields.io/badge/ComfyUI--Whisper-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Whisper)
[![ComfyUI_InvSR](https://img.shields.io/badge/ComfyUI__InvSR-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI_InvSR)
[![ComfyUI-Thera](https://img.shields.io/badge/ComfyUI--Thera-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Thera)
[![ComfyUI-Video-Depth-Anything](https://img.shields.io/badge/ComfyUI--Video--Depth--Anything-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything)
[![ComfyUI-PiperTTS](https://img.shields.io/badge/ComfyUI--PiperTTS-gray?style=flat-square)](https://github.com/yuvraj108c/ComfyUI-PiperTTS)

[![buy-me-coffees](https://i.imgur.com/3MDbAtw.png)](https://www.buymeacoffee.com/yuvraj108cZ)
[![paypal-donation](https://i.imgur.com/w5jjubk.png)](https://paypal.me/yuvraj108c)
---

## Installation
Navigate to the ComfyUI `/custom_nodes` directory
```bash
git clone https://github.com/yuvraj108c/ComfyUI_InvSR
cd ComfyUI_InvSR

# requires diffusers>=0.30.0,<=0.32.2
pip install -r requirements.txt
```

## Usage
- Load [example workflow](workflows/invsr.json) 
- Diffusers model (stabilityai/sd-turbo) will download automatically to `ComfyUI/models/diffusers`
- InvSR model (noise_predictor_sd_turbo_v5.pth) will download automatically to `ComfyUI/models/invsr`
- To deal with large images, e.g, 1k---->4k, set `chopping_size` 256
- If your GPU memory is limited, please set `chopping_batch_size` to 1

## Parameters
- `num_steps`: number of inference steps
- `cfg`: classifier-free guidance scale
- `batch_size`: Controls how many complete images are processed simultaneously
- `chopping_batch_size`: Controls how many patches from the same image are processed simultaneously
- `chopping_size`: Controls the size of patches when splitting large images
- `color_fix`: Method to fix color shift in processed images

## Updates
**28 April 2025**
- Update diffusers versions in requirements.txt to fix https://github.com/yuvraj108c/ComfyUI_InvSR/issues/26, https://github.com/yuvraj108c/ComfyUI_InvSR/issues/21, https://github.com/yuvraj108c/ComfyUI_InvSR/issues/15
- Add support for `noise_predictor_sd_turbo_v5_diftune.pth`
  
**03 February 2025**
- Add cfg parameter
- Make image divisible by 16
- Use `mm` to set torch device
  
**31 January 2025**
- Merged https://github.com/yuvraj108c/ComfyUI_InvSR/pull/5 by [wfjsw](https://github.com/wfjsw)
  - Compatibility with `diffusers>=0.28`
  - Massive code refactoring & cleanup

## Citation
```bibtex
@article{yue2024InvSR,
  title={Arbitrary-steps Image Super-resolution via Diffusion Inversion},
  author={Yue, Zongsheng and Kang, Liao and Loy, Chen Change},
  journal = {arXiv preprint arXiv:2412.09013},
  year={2024},
}
```

## License
This project is licensed under [NTU S-Lab License 1.0](LICENSE)

## Acknowledgments
Thanks to [simplepod.ai](https://simplepod.ai/) for providing GPU servers

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=yuvraj108c/ComfyUI_InvSR&type=Date)](https://star-history.com/#yuvraj108c/ComfyUI_InvSR&Date)
