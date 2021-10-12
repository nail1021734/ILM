from functools import partial

from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers import GPT2Config

from data_processor import create_data_loader
from ray import tune
from train_script import train

if __name__ == '__main__':
    train_config = {
        'max_length': 512,
        'tokenizer_name': 'chinese_tokenizer',
        'dataset_size': 40000,
        'save_ckpt_step': 10000,
        'log_step': 500,
        'padding_idx': 0
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
    hyperopt_search = HyperOptSearch(
        metric='loss', mode='min'
    )
    hp_config = {
        'seed': tune.choice(list(range(0, 41))),
        'lr': tune.uniform(3e-5, 3e-4),
        'epoch_num': 10,
        'batch_size': 4,
        'accumulate_step': tune.choice([20, 30, 40]),
        'warm_up_step_rate': tune.uniform(0.01, 0.05)
    }

    tune.run(
        partial(train, train_config=train_config, model_config=model_config,
                data_loader_creater=create_data_loader),
        name='test',
        config=hp_config,
        num_samples=10,
        local_dir='epoch10',
        search_alg=hyperopt_search,
        resources_per_trial={
            'gpu': 1
        }
    )
