from functools import partial

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from transformers import GPT2Config

from utils.config import build_training_config
from utils.data_processor import create_data_loader
from utils.train_script import train

if __name__ == '__main__':
    # Create training config.
    train_config = build_training_config(
        max_length= 512,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=50000,
        log_step=500,
        exp_name='MLM_exp4',
        task='MLM',
        dataset_name='MLM_dataset_v2',
        ray=True
    )
    # Create model config.
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
    # Create hyperparameter config.
    hp_config = {
        'seed': 42,
        'lr': tune.uniform(3e-5, 3e-4),
        'epoch_num': 3,
        'batch_size': 8,
        'accumulate_step': tune.choice([20, 30, 40, 50]),
        'warm_up_step_rate': tune.uniform(0.01, 0.1),
        'weight_decay': tune.uniform(0.01, 0.05)
    }

    tune.run(
        partial(train, train_config=train_config, model_config=model_config,
                data_loader_creater=create_data_loader),
        name='test',
        config=hp_config,
        num_samples=5,
        local_dir='MLMtest',
        search_alg=hyperopt_search,
        resources_per_trial={
            'gpu': 1
        }
    )
