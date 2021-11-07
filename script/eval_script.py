from utils.eval_methods import eval

if __name__ == '__main__':
    print(eval(
        eval_method='perplexity',
        tokenizer_name='chinese_tokenizer_big',
        ckpt_path='checkpoint/test/checkpoint-400.pt',
        max_seq_len=100,
        eval_dataset='MLM_dataset_v3',
        eval_mode='MLM',
    ))
