import os

from transformers import GPT2Config

from data_processor import create_data_loader
from train_script import train

if __name__ == '__main__':
    train_config = {
        'max_length': 512,
        'tokenizer_name': 'chinese_tokenizer',
        'dataset_size': 40000,
        'save_ckpt_step': 10000,
        'log_step': 500,
        'padding_idx': 0,
        'exp_name': 'test'
    }
    model_config = GPT2Config(
        vocab_size=50000,
        n_positions=512,
        n_ctx=512,
        n_embd=768,
        n_head=12,
        n_inner=2048,
        n_layer=12
    )
    hp_config = {
        'seed': 22,
        'lr': 1e-4,
        'epoch_num': 10,
        'batch_size': 4,
        'accumulate_step': 40,
        'warm_up_step_rate': 0.02
    }
    save_path = os.path.join('checkpoint', train_config['exp_name'])
    train(
        hp_config=hp_config,
        train_config=train_config,
        model_config=model_config,
        data_loader_creater=create_data_loader,
        save_path=save_path,
        progress_bar=True
    )
