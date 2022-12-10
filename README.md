# Speech-Dereverberation-and-RIR-Estimation
![CI Tests](https://github.com/jdonley/Speech-Dereverberation-and-RIR-Estimation/actions/workflows/python-package-conda.yml/badge.svg)

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

## Training Models
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

Results are saved in logs under `./lightning_logs/` and progress images are also saved under `./images/`.

### Notes
The installation example assumes that
[cuda 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive) is installed, along with a
suitable [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) package. See the
[available Conda PyTorch install files](https://anaconda.org/pytorch/pytorch/files) for compatible
CUDA Toolkit - cuDNN combinations. Use `nvcc --version` to determine the CUDA Toolkit version.

If using Windows, a good alternative can be to use WSL2. The setup instructions above work *after* installing WSL2 and NVIDIA CUDA with the [instructions here](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl).
