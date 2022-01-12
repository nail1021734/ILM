import os
import pickle

from utils.create_data import mask_dataset

if __name__ == '__main__':
    training_set, test_set = mask_dataset(
        dataset_name='merged_data',
        tokenizer_name='chinese_tokenizer_big',
        max_length=400,
        document_mask_p=0.03,
        sentence_mask_p=0.1,
        word_mask_p=0.1,
        ngram_mask_p=0.5,
        min_ngram_length=2,
        max_ngram_length=6,
    )
    pickle.dump(training_set,
                open(os.path.join('dataset', 'ILM_train_merged.pk'), 'wb'))
    pickle.dump(test_set,
                open(os.path.join('dataset', 'ILM_test_merged.pk'), 'wb'))
