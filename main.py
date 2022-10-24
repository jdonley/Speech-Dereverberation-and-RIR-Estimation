from models.lightning_model import LitAutoEncoder
from datasets.speech_data import LibriSpeechDataloader
import pytorch_lightning as pl

# ===========================================================
# PyTorch Lightning Models
autoencoder = LitAutoEncoder()

# Data Loaders
train_loader = LibriSpeechDataloader("train")
val_loader   = LibriSpeechDataloader("val")
test_loader  = LibriSpeechDataloader("test")

# PyTorch Lightning Train
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)


# ===========================================================
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

# choose your trained nn.Module
y_pred = autoencoder.predict()
