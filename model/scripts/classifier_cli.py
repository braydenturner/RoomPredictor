from lightning.pytorch.cli import LightningCLI
from lightning import LightningModule, LightningDataModule

from shared_utilities import Classifier, RoomPredictorDataModule

# python scripts/classifier_cli.py --config configs/default_config.yaml
if __name__ == '__main__':
    
    cli = LightningCLI(
        model_class=Classifier, 
        datamodule_class=RoomPredictorDataModule, 
        run=False)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
