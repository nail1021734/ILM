import pickle
import re
from random import random, randrange

from datasets import load_dataset
from tqdm import tqdm

from utils.data_processor import (load_LM_data, load_LM_NT_data, load_MLM_data,
                                  load_MLM_data_v2, load_tokenizer)


def mask_article(
    tokenizer,
    article: str,
    document_mask_p: float = 0.03,
    sentence_mask_p: float = 0.045,
    word_mask_p: float = 0.05,
    ngram_mask_p: float = 0.5,
    min_ngram_length: int = 2,
    max_ngram_length: int = 6,
):
    sentence_spliter = re.compile(r'([，,。：,:；;！!？?])')
    article = ''.join(tokenizer.tokenize(article)[:tokenizer.model_max_length])
    mask_w_token = tokenizer.mask_token
    mask_sent_token = '[MASK_S]'
    mask_doc_token = '[MASK_D]'
    ans_end_token = '[ANS]'
    if random() < document_mask_p:
        masked_article = mask_doc_token
        answer = article
        return masked_article, answer + ans_end_token
    else:
        sentences = sentence_spliter.split(article)
        masked_sentences = []
        answer = []
        alphabet = ['，', ',', '。', '：', ':', '；', ';', '！', '!', '？', '?']
        for sent in sentences:
            if sent in alphabet:
                masked_sentences.append(sent)
                continue
            if random() < sentence_mask_p:
                masked_sentences.append(mask_sent_token)
                answer.append(sent)
            else:
                tokenized_sent = tokenizer.tokenize(sent)
                word_idx = 0
                while word_idx < len(tokenized_sent):
                    if random() < word_mask_p:
                        if random() < ngram_mask_p:
                            n = randrange(
                                min_ngram_length,
                                max_ngram_length
                            )
                            masked_sentences.append(mask_w_token)
                            answer.append(
                                ''.join(tokenized_sent[word_idx: word_idx+n]))
                            word_idx += n
                            continue
                        else:
                            masked_sentences.append(mask_w_token)
                            answer.append(tokenized_sent[word_idx])
                    else:
                        masked_sentences.append(tokenized_sent[word_idx])
                    word_idx += 1
        return (''.join(masked_sentences), ans_end_token.join(answer) + ans_end_token)


if __name__ == '__main__':
    tokenizer = load_tokenizer(
        'chinese_tokenizer_big_NT',
        max_length=400
    )
    dataset = load_LM_NT_data()
    train_data_result = []
    for data in tqdm(dataset['train']):
        data['masked_article'], data['answer'] = mask_article(
            tokenizer=tokenizer, article=data['article'])
        train_data_result.append(data)
    pickle.dump(train_data_result, open('dataset/mlm_train_NT.pk', 'wb'))
    test = []
    for data in tqdm(dataset['test']):
        data['masked_article'], data['answer'] = mask_article(
            tokenizer=tokenizer, article=data['article'])
        test.append(data)
    pickle.dump(test, open('dataset/mlm_test_NT.pk', 'wb'))
