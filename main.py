from models.lightning_model import LitAutoEncoder
from datasets.reverb_speech_data import DareDataloader
import pytorch_lightning as pl

# ===========================================================
# PyTorch Lightning Models
autoencoder = LitAutoEncoder()

# Data Loaders
train_loader = DareDataloader("train")
val_loader   = DareDataloader("val")
test_loader  = DareDataloader("test")

# PyTorch Lightning Train
trainer = pl.Trainer(limit_train_batches=4000, max_epochs=10, accelerator="gpu", devices=1, strategy="dp")
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)

# ===========================================================
# PyTorch Lightning Test
trainer.test(model=autoencoder, dataloaders=test_loader, ckpt_path="best")

# ===========================================================
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint)

# choose your trained nn.Module
y_pred = autoencoder.predict()
