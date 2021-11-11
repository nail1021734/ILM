import pickle

from utils.create_data import create_MN_data

if __name__ == '__main__':
    MN_dataset = create_MN_data(
        ckpt_path='checkpoint/MLM_exp10/checkpoint-4086331.pt',
        dataset_name='MLM_dataset_v3',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        p=0.96,
        data_num=50000,
        use_test_data=False,
        mask_range={'max': 0.35, 'min': 0.25},
    )
    pickle.dump(MN_dataset, open('MN_dataset_25~35%.pk', 'wb'))
