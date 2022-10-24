# Speech-Dereverberation-and-RIR-Estimation

## Setup

We'll use Miniconda to create a virtual environment from which the project can be run.
1. Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Update conda: `conda update conda`
3. Setup virtual environment: `conda create -n SpeechDARE python=3.8 anaconda`. Note `SpeechDARE`
can be substituted for any desired virtual environment name.
4. Activate virtual environment: `conda activate SpeechDARE`.
5. Install dependencies. This example assumes that
[cuda 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive) is installed, along with a
suitable [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) package. See the
[available Conda PyTorch install files](https://anaconda.org/pytorch/pytorch/files) for compatible
CUDA Toolkit - cuDNN combinations. Use `nvcc --version` to determine the CUDA Toolkit version and
check the contents of `$CUDA_PATH/include/cudnn_version.h` for the CuDNN version.
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pytorch-lightning -c conda-forge
conda install numpy pyyaml soundfile matplotlib
```
