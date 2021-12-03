from utils.eval_methods import eval

if __name__ == '__main__':
    r_list = [
        range(4080001, 4080001, 80000),
        range(2640000, 4080001, 80000),
        range(2160000, 4000001, 80000),
    ]
    result_v3 = []
    for i in range(11, 14):
        result_v3.append({})
        for j in r_list[i - 11]:
            score = eval(
                eval_method='perplexity',
                tokenizer_name='chinese_tokenizer_big',
                ckpt_path=f'checkpoint/MLM_exp{i}/checkpoint-{j}.pt',
                max_seq_len=512,
                eval_dataset='MLM_dataset_v3',
            )
            print(score)
            result_v3[-1][j] = score
    r_list = [
        # range(80000, 4080001, 80000),
        range(2960000, 4080001, 80000),
        range(2560000, 4000001, 80000),
    ]
    result_v4 = []
    for i in range(12, 14):
        result_v4.append({})
        for j in r_list[i - 12]:
            score = eval(
                eval_method='perplexity',
                tokenizer_name='chinese_tokenizer_big',
                ckpt_path=f'checkpoint/MLM_exp{i}/checkpoint-{j}.pt',
                max_seq_len=512,
                eval_dataset='MLM_dataset_v4',
            )
            print(score)
            result_v4[-1][j] = score
    r_list = [
        # range(80000, 4080001, 80000),
        range(3360000, 4080001, 80000),
        range(2960000, 4000001, 80000),
    ]
    result_v5 = []
    for i in range(12, 14):
        result_v5.append({})
        for j in r_list[i - 12]:
            score = eval(
                eval_method='perplexity',
                tokenizer_name='chinese_tokenizer_big',
                ckpt_path=f'checkpoint/MLM_exp{i}/checkpoint-{j}.pt',
                max_seq_len=512,
                eval_dataset='MLM_dataset_v5',
            )
            print(score)
            result_v5[-1][j] = score
    # score = eval(
    #     eval_method='perplexity',
    #     tokenizer_name='chinese_tokenizer_big',
    #     ckpt_path=f'checkpoint/MLM_exp11/checkpoint-2720000.pt',
    #     max_seq_len=512,
    #     eval_dataset='MLM_dataset_v3',
    # )
    # print(score)
