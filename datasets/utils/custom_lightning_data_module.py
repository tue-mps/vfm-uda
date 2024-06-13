import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids


class CustomLightningDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            root,
            devices,
            batch_size: int,
            img_size: int,
            train_num_workers: int,
            pin_memory: bool = True,
            persistent_workers: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.devices = devices
        if devices != "auto":
            self.devices = _parse_gpu_ids(devices, include_cuda=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_num_workers = train_num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
