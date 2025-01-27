

<div align="center">

# ComfyUI InvSR
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.09013-b31b1b.svg)](https://arxiv.org/abs/2412.09013) 

This project is an unofficial ComfyUI implementation of [InvSR](https://github.com/zsyOAOA/InvSR) (Arbitrary-steps Image Super-resolution via Diffusion Inversion)

<img height="500" src="https://github.com/user-attachments/assets/117d0668-1c72-46f6-a41d-d3c780614fb7" />

</div>

## Installation
Navigate to the ComfyUI `/custom_nodes` directory
```bash
git clone https://github.com/yuvraj108c/ComfyUI_InvSR
cd ComfyUI_InvSR

# Requirements
pip install -e ".[torch]"
pip install -r requirements.txt
```

## Usage
- Load [example workflow](workflows/invsr.json) 
- Diffusers model (stabilityai/sd-turbo) will download automatically to `ComfyUI/diffusers`
- InvSR model (noise_predictor_sd_turbo_v5.pth) will download automatically to `ComfyUI_InvSR/weights`

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
