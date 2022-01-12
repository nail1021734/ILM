from utils.config import build_configs
from utils.train import train

if __name__ == '__main__':
    # Create training config.
    config, model_config = build_configs(
        # Training config.
        max_length=512,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=160000,
        log_step=500,
        exp_name='MLM_exp_merge2',
        dataset_name='load_ILM_merged_data',
        from_pretrained_model='checkpoint/MLM_exp_merge/checkpoint-1200000.pt',

        # Hyperparameters.
        seed=22,
        lr=2e-4,
        epoch_num=30,
        batch_size=4,
        accumulate_step=80,
        warm_up_step_rate=0.0,
        weight_decay=0.01,

        # Model config. (Use to create `GPT2Config` object.)
        model_config={
            'n_positions': 768,
            'n_ctx': 768,
            'n_embd': 768,
            'n_head': 12,
            'n_inner': 2048,
            'n_layer': 12
        })
    train(config=config, model_config=model_config)
