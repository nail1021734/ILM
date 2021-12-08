import pickle

from utils.create_data import create_MN_data

if __name__ == '__main__':
    for i in range(5):
        MN_dataset = create_MN_data(
            ckpt_path='checkpoint/MLM_exp_token/checkpoint-240000.pt',
            dataset_name='MLM_dataset_v4',
            tokenizer_name='chinese_tokenizer_big',
            max_seq_len=512,
            p=0.8,
            data_num=1000,
            mask_strategy='Token',
            use_test_data=True,
            batch_size=32,
            # mask_rate={'max': 0.35, 'min': 0.25},
            mask_rate=0.20,
        )
    # pickle.dump(MN_dataset, open('test.pkl', 'wb'))
