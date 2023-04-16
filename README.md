# Speech-Dereverberation-and-RIR-Estimation
![CI Tests](https://github.com/jdonley/Speech-Dereverberation-and-RIR-Estimation/actions/workflows/python-package-conda.yml/badge.svg)

This repository contains code for the methods, models and results described in "DARE-Net: Speech Dereverberation and Room Impulse Response Estimation".
If you use this in your work, please use the following citation:
```
@article{darenet2022,
  title={{DARE-Net}: Speech Dereverberation And Room Impulse Response Estimation},
  author={Donley, Jacob and Calamia, Paul},
  journal={Stanford University}, month={December}, year={2022}
}
```

## Setup

We'll use Miniconda to create a virtual environment from which the project can be run.
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Run the following to:
    - Update conda
    - Setup virtual environment
    - Activate virtual environment
    - Install dependencies

```
conda update -y conda
conda create -y -n SpeechDARE
conda activate SpeechDARE
conda install -y python=3.8 anaconda
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y pytorch-lightning -c conda-forge
conda install -y numpy pyyaml matplotlib librosa torchmetrics -c conda-forge
pip install soxr rich
```

Check that the GPU can be used by PyTorch:
```
python -c "import torch; print(torch.cuda.is_available())"
```
If the command above returns `false`, you will need to modify the CUDA and PyTorch installation versions to support your system. Otherwise, you will be limited to using the CPU for processing. See [Notes](#notes) for more.

## Dataset Preperation
The [LibriSpeech dataset](https://www.openslr.org/12) and the [MIT RIR Survey dataset](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) are used. The datasets should automatically download and be prepared the first time the models are trained, validated or tested.

## Training and Testing Models
To train a model, create or edit a `config.yaml` file and then pass the config file to `main.py` with the `--config_path` argument.
There are example configs under the `configs` directory. The following should train, validate and test the main PyTorch Lightning model, `SpeechDAREUnet_v2`.
```
python main.py --config_path ./configs/config.yaml
```

You can follow the progress using `tensorboard` by running 
```
tensorboard --logdir ./lightning_logs/
```
and then going to the URL that the `tensorboard` command hosts. For example, http://localhost:6001.

Results are saved in logs under `./lightning_logs/` and progress images are saved under `./images/`.

## Pre-trained Models
Pre-trained PyTorch model checkpoints for reproducing some of the results in the paper can be [downloaded from here](https://drive.google.com/drive/folders/1XMINM6dUyXIipjLHcRicEyPj3d6K4lgY?usp=sharing).

## Notes
The installation example assumes that
[cuda 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive) is installed, along with a
suitable [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) package. See the
[available Conda PyTorch install files](https://anaconda.org/pytorch/pytorch/files) for compatible
CUDA Toolkit - cuDNN combinations. Use `nvcc --version` to determine the CUDA Toolkit version.

If using Windows, a good alternative can be to use WSL2. The setup instructions above work *after* installing WSL2 and NVIDIA CUDA with the [instructions here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
