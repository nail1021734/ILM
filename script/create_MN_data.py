import pickle

from utils.create_data import create_MN_data

if __name__ == '__main__':
    MN_dataset = create_MN_data(
        ckpt_path='checkpoint/MLM_exp11/checkpoint-2160000.pt',
        dataset_name='MLM_dataset_v3',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        k=40,
        data_num=50000,
        mask_strategy='Sentence',
        use_test_data=True,
        batch_size=32,
        mask_range={'max': 0.35, 'min': 0.25},
    )
    pickle.dump(MN_dataset, open('test.pkl', 'wb'))
