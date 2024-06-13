# ---------------------------------------------------------------
# Copyright (c) 2024 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------


import logging

import torch
from gitignore_parser import parse_gitignore
from lightning import LightningModule, seed_everything
from lightning.pytorch import cli

from datasets.utils.custom_lightning_data_module import CustomLightningDataModule


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", type=bool, default=False)
        parser.add_argument("--root", type=str)
        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")
        parser.link_arguments("trainer.devices", "data.init_args.devices")
        parser.link_arguments("data.init_args.batch_size", "model.init_args.batch_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments("data.init_args.img_size", "model.init_args.network.init_args.img_size")

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):  # type: ignore
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(  # type: ignore
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        if self.config[self.config["subcommand"]]["compile"]:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = False
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)  # type: ignore


def cli_main():
    # only use "strategy": "ddp_find_unused_parameters_true" during testing:
    # Warning: find_unused_parameters=True was specified in DDP constructor,
    # but did not find any unused parameters in the forward pass.
    # This flag results in an extra traversal of the autograd graph every iteration,
    # which can adversely affect performance.
    # If your model indeed never has any unused parameters in the forward pass,
    # consider turning this flag off.
    # Note that this warning may be a false positive,
    # if your model has flow control causing later iterations to have unused parameters.
    LightningCLI(
        LightningModule,
        CustomLightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults={
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "strategy": "ddp_find_unused_parameters_true"
        },
    )


if __name__ == "__main__":
    cli_main()
