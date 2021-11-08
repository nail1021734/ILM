import os
import pickle

from utils.create_data import mask_dataset

if __name__ == '__main__':
    training_set, test_set = mask_dataset(
        dataset_name='LM_NT_data',
        tokenizer_name='chinese_tokenizer_big_NT',
        max_length=400,
    )
    pickle.dump(training_set, open(
        os.path.join('dataset', 'mlm_train_NT.pk'), 'wb'))
    pickle.dump(test_set, open(
        os.path.join('dataset', 'mlm_test_NT.pk'), 'wb'))
