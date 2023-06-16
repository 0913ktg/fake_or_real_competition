# code for train pipeline
import torch
import argparse
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from config.config import *
from src.utils import dotdict
from src.experiments.experiment import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="model1",
        required=True,
        type=str,
        help="model name that you want to train",
    )
    parser.add_argument(
        "--dataset",
        default="dataset1",
        required=True,
        type=str,
        help="dataset name that you use for train",
    )

    return parser.parse_args()


def train():
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset, val_dataset, test_dataset = create_datasets()
    train_loader, val_loader, test_loader = initiate_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        env=env,
    )

    # model
    model = instantiate_model(env)
    model = model.to(device)

    # pytorch_lightning setting
    callbacks = []

    gpus = -1 if torch.cuda.is_available() else 0
    logger = MLFlowLogger(experiment_name="sample_experiment_name", save_dir="./mlruns")

    trainer = pl.Trainer(
        max_epochs=100,
        #  max_steps=mc['max_steps'],
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        progress_bar_refresh_rate=1,
        gpus=gpus,
        callbacks=callbacks,
        logger=logger,
        devices=1,
        strategy="ddp",
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    # dictionary for configure, it can be accessed .
    config = dotdict()

    # config.py에 있는 env update(실험 관련 변수)
    config.update(env)
    # add args to config
    config.update(vars(args))
    # config.py에 {특정 모델 이름}_config에 저장된 정보
    config.model_config = eval(f"{args.model}_config")
    # config.py에 {특정 데이터셋 이름}_config에 저장된 정보
    config.dataset_config = eval(f"{args.dataset}_config")
