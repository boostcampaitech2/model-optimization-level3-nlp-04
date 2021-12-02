"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
import os
import random
from datetime import datetime
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorly as tl
import wandb
import yaml
from transformers import is_torch_available

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.torch_utils import check_runtime, model_info, decompose


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    args,
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    wandb.init(project=args.project_name,
               name=args.run_name,
               reinit=True,
               )

    # save model_config, data_config
    with open(os.path.join(log_dir, "data.yml"), "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, "model.yml"), "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(
            torch.load(model_path, map_location=device)
        )

    if args.td:
        # switch to the PyTorch backend
        try:
            tl.set_backend('pytorch')
        except:
            tl.set_backend('pytorch')

        for name, param in model_instance.model.named_modules():
            if isinstance(param, nn.Conv2d):
                param.register_buffer('rank', torch.Tensor([0.5, 0.5]))  # rank in, out

        model_instance.model = decompose(model_instance.model)

    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Create optimizer, scheduler, criterion
    try:
        if data_config['OPTIMIZER_NAME'] == 'SGD':
            optimizer = torch.optim.SGD(
                model_instance.model.parameters(), lr=data_config["LR"], momentum=data_config['MOMENTUM']
            )
        elif data_config['OPTIMIZER_NAME'] == 'Adam':
            optimizer = torch.optim.Adam(
                model_instance.model.parameters(), lr=data_config['LR'],
                betas=(data_config['BETA_1'], data_config['BETA_2'])
            )
    except:
        optimizer = torch.optim.SGD(
            model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
        )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=data_config['INIT_LR'],
        steps_per_epoch=len(train_dl),
        epochs=data_config["EPOCHS"],
        pct_start=0.05,
    )
    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )

    wandb.join()

    return test_loss, test_f1, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model",
        default="configs/model/base_model.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--data",
        default="configs/data/base_data.yaml",
        type=str, help="data config"
    )
    parser.add_argument("--project_name", default="raki_base_model_test", type=str, help="wandb project name")
    parser.add_argument("--run_name", default="no_cutmix", type=str, help="wandb run name")
    parser.add_argument("--td", default=False, type=bool, help="whether use Tucker Decomposition")

    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    data_config["DATA_PATH"] = os.environ.get("SM_CHANNEL_TRAIN", data_config["DATA_PATH"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_dir = os.environ.get("SM_MODEL_DIR", os.path.join("exp", 'latest'))

    if os.path.exists(log_dir): 
        modified = datetime.fromtimestamp(os.path.getmtime(log_dir + '/best.pt'))
        new_log_dir = os.path.dirname(log_dir) + '/' + modified.strftime("%Y-%m-%d_%H-%M-%S")
        os.rename(log_dir, new_log_dir)

    os.makedirs(log_dir, exist_ok=True)

    # set_seed(42)

    test_loss, test_f1, test_acc = train(
        args=args,
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )

