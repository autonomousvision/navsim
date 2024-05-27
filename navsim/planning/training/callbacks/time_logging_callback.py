from typing import Any, Optional
import time

import pytorch_lightning as pl


class TimeLoggingCallback(pl.Callback):
    """Simple lightning callback to log training time."""

    def __init__(self) -> None:
        pass

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.val_start = time.time()

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        lightning_module.log_dict(
            {
                "time_eval": time.time() - self.val_start,
                "step": lightning_module.current_epoch,
            }
        )

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.test_start = time.time()

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        lightning_module.log_dict(
            {
                "time_test": time.time() - self.test_start,
                "step": lightning_module.current_epoch,
            }
        )

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        self.train_start = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""
        lightning_module.log_dict(
            {
                "time_epoch": time.time() - self.train_start,
                "step": lightning_module.current_epoch,
            }
        )
