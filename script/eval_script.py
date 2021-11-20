from utils.eval_methods import eval

if __name__ == '__main__':
    result = {}
    for i in range(3200000, 3680000, 80000):
        score = eval(
            eval_method='perplexity',
            tokenizer_name='chinese_tokenizer_big',
            ckpt_path=f'checkpoint/MLM_exp11/checkpoint-{i}.pt',
            max_seq_len=512,
            eval_dataset='MLM_dataset_v3',
        )
        print(score)
        result[i] = score
    for k, v in result.items():
        print(k, ': ', v)
    # score = eval(
    #     eval_method='perplexity',
    #     tokenizer_name='chinese_tokenizer_big',
    #     ckpt_path=f'checkpoint/MLM_exp11/checkpoint-2720000.pt',
    #     max_seq_len=512,
    #     eval_dataset='MLM_dataset_v3',
    # )
    # print(score)
