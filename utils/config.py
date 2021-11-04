import os

from _path import ROOT_PATH
from utils.data_processor import (load_LM_data, load_MLM_data, load_MLM_NT_data,
                                  load_MLM_data_v2, load_LM_NT_data, load_MLM_data_v3, load_tokenizer)


def build_training_config(
    max_length: int,
    tokenizer_name: str,
    save_ckpt_step: int,
    log_step: int,
    dataset_name: str,
    task: str = None,
    exp_name: str = None,
    ray: bool = False,
):
    # Load dataset to count amount of data.
    # (This information will use in calcuating warm up step.)
    if dataset_name == 'Taiwan_news_dataset':
        dataset = load_LM_data()
        # If not specify task then default use LM task.
        if task == None:
            task = 'LM'
    elif dataset_name == 'MLM_dataset':
        dataset = load_MLM_data()
        # If not specify task then default use MLM task.
        if task == None:
            task = 'MLM'
    elif dataset_name == 'MLM_dataset_v2':
        dataset = load_MLM_data_v2()
        # If not specify task then default use MLM task.
        if task == None:
            task = 'MLM'
    elif dataset_name == 'MLM_dataset_v3':
        dataset = load_MLM_data_v3()
        # If not specify task then default use MLM task.
        if task == None:
            task = 'MLM'
    elif dataset_name == 'MLM_NT_data':
        dataset = load_MLM_NT_data()
        # If not specify task then default use MLM task.
        if task == None:
            task = 'MLM'
    elif dataset_name == 'LM_NT_data':
        dataset = load_LM_NT_data()
        # If not specify task then default use MLM task.
        if task == None:
            task = 'LM'
    else:
        raise Exception('Dataset name not exist.')

    # If use manual train than create save path to save checkpoint.
    if ray:
        save_path = None
    else:
        if not exp_name:
            raise Exception(
                'Must have `exp_name` in training config to ' +
                'create directory to save checkpoint.'
            )
        save_path = os.path.join('checkpoint', exp_name)

    # Load tokenizer to get padding token id.
    tokenizer = load_tokenizer(tokenizer_name, max_length=max_length)

    # Return training config.
    return {
        'max_length': max_length,
        'tokenizer_name': tokenizer_name,
        'dataset_size': len(dataset['train']),
        'save_ckpt_step': save_ckpt_step,
        'log_step': log_step,
        'padding_idx': tokenizer.pad_token_id,
        'exp_name': exp_name,
        'task': task,
        'dataset_name': dataset_name,
        'save_path': save_path
    }
