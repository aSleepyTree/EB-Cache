# Code for：FEB-Cache: Frequency‑Guided Exposure Bias Reduction for Enhancing Diffusion Transformer Caching


# 🔧 Dependencies and Sampling

- Python >= 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13.0+cu11.7](https://pytorch.org/)
```bash
conda create -n FORA python=3.9
conda activate FORA
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

## Cache Table
cache_table.npy is an example generated on ImageNet, and you can directly use it or regenerate a new one.

## DiT sampling
- To sample for single ImageNet class with conditional guidance strength 1.5, with caching frequency 3, with output image size 512 and with DDIM steps 250
```bash
python src/sample.py --save-cache 'boost_infer_static' --cache-subtype 'default' --cache-threshold '3' --image-size 512 --seed 1 --cfg-scale 1.5 --num-sampling-steps 250
```
- To sample for entire ImageNetdataset and save the output in samples folder
```bash
 torchrun --nnodes=1 --nproc_per_node=4 src/sample_ddp.py --num-fid-samples 50000 --save-cache 'boost_infer_static' --cache-subtype 'default' --cache-threshold '3' --image-size 256 --per-proc-batch-size 4 --sample-dir 'samples' --cfg-scale 1.5 --num-sampling-steps 250
```


This rope is based on [FORA](https://github.com/prathebaselva/FORA) and [Delta-DiT](https://arxiv.org/abs/2406.01125). Thanks for their nice work.