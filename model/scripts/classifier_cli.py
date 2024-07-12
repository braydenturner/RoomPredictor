from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from shared_utilities import *

if __name__ == '__main__':

    cli = LightningCLI(
        model_class=Classifier,
        datamodule_class=RoomPredictorDataModule,
        run=False,
        save_config_callback=None,
        seed_everything_default=411994,
        trainer_defaults={
            "max_epochs": 100,
            "accelerator": "cpu",
            "callbacks": [ModelCheckpoint(monitor="val_acc", mode="max")]
        }
    )

    model = Classifier(num_hidden=cli.model.num_hidden, num_outputs=5, lr=cli.model.lr, dropout=cli.model.dropout)
    datamodule = RoomPredictorDataModule(file=cli.datamodule.file)

    cli.trainer.fit(model, datamodule=datamodule)
    cli.trainer.test(model, datamodule=datamodule)
