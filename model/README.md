# Classifier




## PytorchLightning

PyTorch Lightning is a lightweight PyTorch wrapper that simplifies the process of writing PyTorch code by decoupling the research code (model, training, validation) from engineering code (GPU, TPU, multi-GPU, etc.). It’s designed to scale and make your models easier to reproduce and debug.

Basic Components of PyTorch Lightning
* LightningModule – Defines the model, training, validation, and test logic.
* LightningDataModule – Organizes data loading and preprocessing.
* Trainer – Handles training loops, hardware acceleration, logging, etc.


### Model
The model file looks like this

```python
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
```
        
        
### Data
The data module file looks something liek this

```python
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def prepare_data(self):
        datasets.MNIST(root='./data', train=True, download=True)
        datasets.MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST(root='./data', train=True, transform=self.transform)
        self.mnist_val = datasets.MNIST(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)
```


### Trainer
The training file looks like this

```python
model = MNISTModel()
data_module = MNISTDataModule()

trainer = Trainer(max_epochs=5)
trainer.fit(model, datamodule=data_module)
```

### Callbacks

Callbacks can be used for checkpointing and logging

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')
trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=10)
trainer.fit(model, datamodule=data_module)
```



### Configs

Print a configuration to have as reference
`python main.py fit --print_config > config.yaml`

Modify the config to your liking - you can remove all default arguments
`nano config.yaml`

Fit your model using the edited configuration
`python main.py fit --config config.yaml`



Pieces of each part look like this
```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
// Tensorboard logger
  logger: 
    class_path: lightning.pytorch.loggers.TensorBoardLogger
    init_args:
      save_dir: "logs"
      name: "classifier"
// Callbacks for model checkpoint and model summary
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: "val_accuracy"
        mode: "max"
    - class_path: lightning.pytorch.callbacks.ModelSummary
      init_args:
        max_depth: -1


model:
  input_size: 784
  hidden_size: 128
  output_size: 10
  learning_rate: 0.001



data:
  batch_size: 64
  dataset: MNIST
  data_path: ./data
  ```