from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

def getProgressBar(config):
    return RichProgressBar(theme=RichProgressBarTheme(**config['RichProgressBarTheme']))
