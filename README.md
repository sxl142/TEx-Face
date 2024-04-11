# TEx-Face
Controllable 3D Face Generation with Conditional Style Code Diffusion (AAAI2024)

[[Project Page](https://sxl142.github.io/TEx-Face/) | [ArXiv](https://arxiv.org/abs/2312.13941) | [Sup](/docs/static/pdfs/TEx-sup.pdf)]


## Installation
```bash
conda create -n texface python=3.8 -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## pretrained model
### 1. Download [EG3D](https://drive.google.com/file/d/1ZAJxfEFbOypRMyCCRA4LfTeIbkOnZi_m/view?usp=drive_link) and [ir_se50](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view), put them into 'inversion/pretrained'

### 2. Download [Re-PoI](https://drive.google.com/file/d/1RAJ5-B-92Ygg5aR-T1cx0U06uTsSyBqo/view?usp=sharing).

### 3. Download [SCD](https://drive.google.com/file/d/10lhpORq2g_K7WVafxdRt0CVBiOQxoq90/view?usp=sharing) and put it into 'generation/checkpoints'.

## Test Inversion
```bash
cd inversion
python scripts/test_celeba.py ./checkpoints.pt
```
## Test Generation
```bash
cd generation
python scripts/infer.py 
python scripts/gen_videos.py
```