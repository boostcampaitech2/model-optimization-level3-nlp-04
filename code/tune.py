"""Tune Model.
- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info, check_runtime
from src.trainer import TorchTrainer, count_model_params
from typing import Any, Dict, List, Tuple
from optuna.pruners import HyperbandPruner
from subprocess import _args_from_interpreter_flags
import argparse

EPOCH = 100
DATA_PATH = "/opt/ml/data"  # type your data path here that contains test, train and val directories
RESULT_MODEL_PATH = "./result_model.pt" # result model will be saved in this path


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=1, high=1, step=50) # origin : (50, 50, 50)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=16, high=32, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size,
    }


def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    module_info = {}

    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2  # 5(224 example): 224, 112, 56, 28, 14, 7

    modules = {
        # 'm' : [Conv], (repeat_start, repeat_end), (channels_low, channels_high, channels_step),
        # [Activation], (kernel_low, kernel_high, kernel_step),
        # (c2_low, c2_high, c2_step), (t2_low, t2_high),
        # (t3_low, t3_high, t3_step, t3_round), (c3_low, c3_high, c3_step),
        # (kernel3_low, kernel3_high, kernel3_step), [se3], [hs3],
        'm1': [
            ["Conv", "DWConv"], (1, 3), (16, 64, 16),
            ["ReLU", "Hardswish"], (3, 3, 3),
            (0, 0, 1), (0, 0),
            (0, 0, 1, 1), (0, 0, 1),
            (0, 0, 1), [0], [0],
        ],
        'm2': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 128, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (16, 32, 16), (1, 4),
            (1.0, 6.0, 0.1, 1), (16, 40, 8),
            (3, 5, 2), [0, 1], [0, 1]
        ],
        'm3': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 128, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (8, 32, 8), (1, 8),
            (1.0, 6.0, 0.1, 1), (8, 40, 8),
            (3, 5, 2), [0, 1], [0, 1]
        ],
        'm4': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 256, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (8, 64, 8), (1, 8),
            (1.0, 6.0, 0.1, 1), (8, 80, 8),
            (3, 5, 2), [0, 1], [0, 1]
        ],
        'm5': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 256, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (16, 128, 16), (1, 8),
            (1.0, 6.0, 0.1, 1), (16, 80, 16),
            (3, 5, 2), [0, 1], [0, 1]
        ],
        'm6': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 512, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (16, 128, 16), (1, 8),
            (1.0, 6.0, 0.1, 1), (16, 160, 16),
            (3, 5, 2), [0, 1], [0, 1]
        ],
        'm7': [
            ["Conv", "DWConv", "InvertedResidualv2", "InvertedResidualv3", "Pass"], (1, 5), (16, 1024, 16),
            ["ReLU", "Hardswish"], (1, 5, 2),
            (16, 160, 16), (1, 8),
            (1.0, 6.0, 0.1, 1), (8, 160, 8),
            (3, 5, 2), [0, 1], [0, 1]
        ],

    }

    for module, component in modules.items():
        conv, repeat, channels, activation, kernel, c2, t2, t3, c3, kernel3, se3, hs3 = component
        m = trial.suggest_categorical(module, conv)
        m_repeat = trial.suggest_int(module+'/repeat', repeat[0], repeat[1])
        m_out_channel = trial.suggest_int(module+'/out_channels', low=channels[0], high=channels[1], step=channels[2])
        m_stride = trial.suggest_int(module+'/stride', low=1, high=UPPER_STRIDE)
        if not(int(module[1:]) & 1) and n_stride == int(module[1:]) // 2 - 1:
            m_stride = 2

        m_kernel = trial.suggest_int(module+'/kernel_size', low=kernel[0], high=kernel[1], step=kernel[2])
        m_v2_c = trial.suggest_int(module+'/v2_c', low=c2[0], high=c2[1], step=c2[2])
        m_v2_t = trial.suggest_int(module+'/v2_t', low=t2[0], high=t2[1])
        m_v3_t = round(trial.suggest_float(module+'/v3_t', low=t3[0], high=t3[1], step=t3[2]), t3[3])
        m_v3_c = trial.suggest_int(module+'/v3_c', low=c3[0], high=c3[1], step=8)
        m_kernel3 = trial.suggest_int(module+'/kernel3_size', low=kernel3[0], high=kernel3[1], step=kernel3[2])
        m_v3_se = trial.suggest_categorical(module+'/v3_se', se3)
        m_v3_hs = trial.suggest_categorical(module+'/v3_hs', hs3)

        m_activation = trial.suggest_categorical(module+'/activation', activation)
        if m == "Conv":
            # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
            m_args = [m_out_channel, m_kernel, m_stride, None, 1, m_activation]
        elif m == "DWConv":
            # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
            m_args = [m_out_channel, m_kernel, m_stride, None, m_activation]
        elif m == "InvertedResidualv2":
            m_args = [m_v2_c, m_v2_t, m_stride]
        elif m == "InvertedResidualv3":
            m_args = [m_kernel3, m_v3_t, m_v3_c, m_v3_se, m_v3_hs, m_stride]

        if not m == "Pass":
            if m_stride == 2:
                n_stride += 1
                if n_stride >= MAX_NUM_STRIDE:
                    UPPER_STRIDE = 1
            model.append([m_repeat, m, m_args])

        module_info[module] = {"type": m, "repeat": m_repeat, "stride": m_stride}

    # last layer
    last_dim = trial.suggest_int("last_dim", low=128, high=1024, step=128)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [6, 1, 1, None, 1, None]])

    return model, module_info


def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.
    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    model_config: Dict[str, Any] = {}
    model_config["input_channel"] = 3
    # img_size = trial.suggest_categorical("img_size", [32, 64, 128])
    img_size = 32
    model_config["INPUT_SIZE"] = [img_size, img_size]
    model_config["depth_multiple"] = trial.suggest_categorical(
        "depth_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["width_multiple"] = trial.suggest_categorical(
        "width_multiple", [0.25, 0.5, 0.75, 1.0]
    )
    model_config["backbone"], module_info = search_model(trial)
    hyperparams = search_hyperparam(trial)

    model = Model(model_config, verbose=True)
    model.to(device)
    model.model.to(device)

    # check ./data_configs/data.yaml for config information
    data_config: Dict[str, Any] = {}
    data_config["DATA_PATH"] = DATA_PATH
    data_config["DATASET"] = "TACO"
    data_config["AUG_TRAIN"] = "randaugment_train"
    data_config["AUG_TEST"] = "simple_augment_test"
    data_config["AUG_TRAIN_PARAMS"] = {
        "n_select": hyperparams["n_select"],
    }
    data_config["AUG_TEST_PARAMS"] = None
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["VAL_RATIO"] = 0.8
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    wandb.init(project=)
    mean_time = check_runtime(
        model.model,
        [model_config["input_channel"]] + model_config["INPUT_SIZE"],
        device,
    )
    model_info(model, verbose=True)
    train_loader, val_loader, test_loader = create_dataloader(data_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["EPOCHS"],
        pct_start=0.05,
    )

    trainer = TorchTrainer(
        model,
        criterion,
        optimizer,
        scheduler,
        device=device,
        verbose=1,
        model_path=RESULT_MODEL_PATH,
    )
    trainer.train(train_loader, hyperparams["EPOCHS"], val_dataloader=val_loader)
    loss, f1_score, acc_percent = trainer.test(model, test_dataloader=val_loader)
    params_nums = count_model_params(model)

    model_info(model, verbose=True)
    return f1_score, params_nums, mean_time


def get_best_trial_with_condition(optuna_study: optuna.study.Study) -> Dict[str, Any]:
    """Get best trial that satisfies the minimum condition(e.g. accuracy > 0.8).
    Args:
        study : Optuna study object to get trial.
    Returns:
        best_trial : Best trial that satisfies condition.
    """
    df = optuna_study.trials_dataframe().rename(
        columns={
            "values_0": "acc_percent",
            "values_1": "params_nums",
            "values_2": "mean_time",
        }
    )
    ## minimum condition : accuracy >= threshold
    threshold = 0.7
    minimum_cond = df.acc_percent >= threshold

    if minimum_cond.any():
        df_min_cond = df.loc[minimum_cond]
        ## get the best trial idx with lowest parameter numbers
        best_idx = df_min_cond.loc[
            df_min_cond.params_nums == df_min_cond.params_nums.min()
        ].acc_percent.idxmax()

        best_trial_ = optuna_study.trials[best_idx]
        print("Best trial which satisfies the condition")
        print(df.loc[best_idx])
    else:
        print("No trials satisfies minimum condition")
        best_trial_ = None

    return best_trial_


def tune(gpu_id, storage: str = None):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler()
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None
    study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        study_name="automl101",
        sampler=sampler,
        storage=rdb_storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, device), n_trials=3) # origin 500

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)
    print(best_trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="Optuna database storage path.")
    args = parser.parse_args()
    tune(args.gpu, storage=args.storage if args.storage != "" else None)
