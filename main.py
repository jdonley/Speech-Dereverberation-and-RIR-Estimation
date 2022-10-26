from models.lightning_model import LitAutoEncoder
from datasets.reverb_speech_data import DareDataloader
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# ===========================================================
# PyTorch Lightning Models
autoencoder = LitAutoEncoder()

# Data Loaders
train_loader = DareDataloader("train",batch_size=128)
val_loader   = DareDataloader("val",  batch_size=128)
test_loader  = DareDataloader("test", batch_size=128)

# PyTorch Lightning Train
trainer = pl.Trainer(
    limit_train_batches=17,
    limit_val_batches=2,
    limit_test_batches=2,
    max_epochs=100,
    log_every_n_steps=10,
    accelerator="gpu",
    devices=1,
    strategy="dp"
    )

trainer.fit(
    model=autoencoder,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
    )

# ===========================================================
# PyTorch Lightning Test
trainer.test(
    model=autoencoder,
    dataloaders=test_loader,
    ckpt_path="best"
    )

# ===========================================================
autoencoder = LitAutoEncoder.load_from_checkpoint(str(trainer.tested_ckpt_path))
autoencoder.encoder.eval()
autoencoder.decoder.eval()

# choose your trained nn.Module
y_pred = autoencoder.predict(test_loader.dataset[0][0][None,:].to('cpu'))

plt.plot(y_pred.detach().numpy().T)
plt.show()

print("done")