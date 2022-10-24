import torchaudio
from pathlib import Path
from utils import *

data_ls = torchaudio.datasets.LIBRISPEECH(
    Path(getConfig()['datasets_path']),
    download=True
    )

print(data_ls)
