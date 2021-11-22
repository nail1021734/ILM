import os
import pickle

from utils.create_data import mask_dataset

if __name__ == '__main__':
    training_set, test_set = mask_dataset(
        dataset_name='Taiwan_news_dataset',
        tokenizer_name='chinese_tokenizer_big',
        max_length=400,
    )
    pickle.dump(training_set, open(
        os.path.join('dataset', 'mlm_train_v4.pk'), 'wb'))
    pickle.dump(test_set, open(
        os.path.join('dataset', 'mlm_test_v4.pk'), 'wb'))
