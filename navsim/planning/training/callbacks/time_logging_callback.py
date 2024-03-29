import time
from typing import Any, Optional

import pytorch_lightning as pl


class TimeLoggingCallback(pl.Callback):
    def __init__(self) -> None:
        pass

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        self.val_start = time.time()

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        lightning_module.log_dict(
            {
                'time_eval': time.time() - self.val_start,
                'step': lightning_module.current_epoch,
            }
        )

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        self.test_start = time.time()

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        lightning_module.log_dict(
            {
                'time_test': time.time() - self.test_start,
                'step': lightning_module.current_epoch,
            }
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        self.train_start = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        lightning_module.log_dict(
            {
                'time_epoch': time.time() - self.train_start,
                'step': lightning_module.current_epoch,
            }
        )
