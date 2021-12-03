import os
import pickle

from utils.create_data import mask_dataset

if __name__ == '__main__':
    training_set, test_set = mask_dataset(
        dataset_name='Taiwan_news_dataset',
        tokenizer_name='chinese_tokenizer_big',
        max_length=400,
        document_mask_p=0.03,
        sentence_mask_p=0.045,
        word_mask_p=0.05,
        ngram_mask_p=0.5,
        min_ngram_length=2,
        max_ngram_length=6,
    )
    pickle.dump(training_set, open(
        os.path.join('dataset', 'mlm_train5.pk'), 'wb'))
    pickle.dump(test_set, open(
        os.path.join('dataset', 'mlm_test5.pk'), 'wb'))
