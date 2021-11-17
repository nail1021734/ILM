import os
from typing import Dict

from utils.data_processor import load_dataset_by_name, load_tokenizer


def build_configs(
    # Training config.
    max_length: int,
    tokenizer_name: str,
    save_ckpt_step: int,
    log_step: int,
    exp_name: str,
    dataset_name: str,

    # Hyperparameters.
    seed: int,
    lr: float,
    epoch_num: int,
    batch_size: int,
    accumulate_step: int,
    warm_up_step_rate: float,
    weight_decay: float,

    # Model config. (Use to create `GPT2Config` object.)
    model_config: Dict,

    # If train from pretrained model. (Default None.)
    from_pretrained_model: str = None,
):
    # Load dataset to count amount of data.
    # (This information will use in calcuating warm up step.)
    dataset = load_dataset_by_name(dataset_name=dataset_name)

    # If use manual train than create save path to save checkpoint.
    if not exp_name:
        raise Exception(
            'Must have `exp_name` in training config to ' +
            'create directory to save checkpoint.'
        )
    save_path = os.path.join('checkpoint', exp_name)

    # Load tokenizer to get padding token id.
    tokenizer = load_tokenizer(tokenizer_name, max_length=max_length)


    # Get pretrained model optimizer path.
    if from_pretrained_model:
        dir_path = os.path.dirname(from_pretrained_model)
        ckpt_step = os.path.basename(from_pretrained_model).split('-')[-1]
        optimizer_path = os.path.join(dir_path, f'optimizer-{ckpt_step}')
    else:
        optimizer_path = None

    # Automaticly infill `vocab_size` parameter in model config.
    model_config['vocab_size'] = len(tokenizer.get_vocab())

    # Return training config.
    return (
        {
            # Training config.
            'max_length': max_length,
            'tokenizer_name': tokenizer_name,
            'dataset_size': len(dataset['train']),
            'save_ckpt_step': save_ckpt_step,
            'log_step': log_step,
            'padding_idx': tokenizer.pad_token_id,
            'exp_name': exp_name,
            'dataset_name': dataset_name,
            'save_path': save_path,
            'model_config': model_config,
            'ckpt_path': from_pretrained_model,
            'optimizer_path': optimizer_path,

            # Hyperparameters.
            'seed': seed,
            'lr': lr,
            'epoch_num': epoch_num,
            'batch_size': batch_size,
            'accumulate_step': accumulate_step,
            'warm_up_step_rate': warm_up_step_rate,
            'weight_decay': weight_decay,
        },
        model_config
    )
