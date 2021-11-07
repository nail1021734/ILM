from utils.config import build_configs
from utils.data_processor import create_data_loader
from utils.train import train
from utils.data_processor import load_tokenizer

if __name__ == '__main__':
    # Create training config.
    config, model_config = build_configs(
        # Training config.
        max_length=128,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=100,
        log_step=500,
        exp_name='test',
        dataset_name='MLM_dataset_v3',

        # Hyperparameters.
        seed=22,
        lr=2e-4,
        epoch_num=30,
        batch_size=8,
        accumulate_step=40,
        warm_up_step_rate=0.02,
        weight_decay=0.01,

        # Model config. (Use to create `GPT2Config` object.)
        model_config={
            'n_positions': 128,
            'n_ctx': 128,
            'n_embd': 128,
            'n_head': 8,
            'n_inner': 2048,
            'n_layer': 3
        }
    )
    train(
        config=config,
        model_config=model_config
    )
