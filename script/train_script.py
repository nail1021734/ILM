from utils.config import build_configs
from utils.train import train

if __name__ == '__main__':
    # Create training config.
    config, model_config = build_configs(
        # Training config.
        max_length=512,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=80000,
        log_step=500,
        exp_name='train_from_exp10_reduce_lr',
        dataset_name='MLM_dataset_v3',
        from_pretrained_model='checkpoint/MLM_exp10/checkpoint-4086331.pt',

        # Hyperparameters.
        seed=22,
        lr=2e-5,
        epoch_num=30,
        batch_size=8,
        accumulate_step=40,
        warm_up_step_rate=0.00,
        weight_decay=0.01,

        # Model config. (Use to create `GPT2Config` object.)
        model_config={
            'n_positions': 768,
            'n_ctx': 768,
            'n_embd': 768,
            'n_head': 12,
            'n_inner': 2048,
            'n_layer': 12
        }
    )
    train(
        config=config,
        model_config=model_config
    )
