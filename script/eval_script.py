from utils.eval_methods import eval

if __name__ == '__main__':
    result = {}
    for i in range(3200000, 4086331, 80000):
        score = eval(
            eval_method='perplexity',
            tokenizer_name='chinese_tokenizer_big',
            ckpt_path=f'checkpoint/MLM_exp10/checkpoint-{i}.pt',
            max_seq_len=512,
            eval_dataset='MLM_dataset_v3',
        )
        print(score)
        result[i] = score
    for k, v in result.items():
        print(k, ': ',v)