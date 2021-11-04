from transformers import GPT2Config

from utils.config import build_training_config
from utils.data_processor import create_data_loader
from utils.train_script import train
from utils.data_processor import load_tokenizer

if __name__ == '__main__':
    # Create training config.
    train_config = build_training_config(
        max_length=512,
        tokenizer_name='chinese_tokenizer_big',
        save_ckpt_step=80000,
        log_step=500,
        exp_name='LM_exp2',
        task='LM',
        dataset_name='Taiwan_news_dataset',
        ray=False
    )
    tokenizer = load_tokenizer(
        tokenizer_name=train_config['tokenizer_name'],
        max_length=train_config['max_length']
    )
    # Create model config.
    model_config = GPT2Config(
        vocab_size=len(tokenizer.get_vocab()),
        n_positions=768,
        n_ctx=768,
        n_embd=768,
        n_head=12,
        n_inner=2048,
        n_layer=12
    )
    # Create hyperparameter config.
    hp_config = {
        'seed': 22,
        'lr': 2e-4,
        'epoch_num': 30,
        'batch_size': 8,
        'accumulate_step': 40,
        'warm_up_step_rate': 0.02,
        'weight_decay': 0.01
    }
    train(
        hp_config=hp_config,
        train_config=train_config,
        model_config=model_config,
        data_loader_creater=create_data_loader,
        progress_bar=True
    )
