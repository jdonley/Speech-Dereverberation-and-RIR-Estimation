from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme

def getProgressBar(config):
    return RichProgressBar(**config['RichProgressBar'],theme=RichProgressBarTheme(**config['RichProgressBarTheme']))
