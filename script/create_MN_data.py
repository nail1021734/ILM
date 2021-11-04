from typing import Iterable
from utils.data_processor import (
    load_dataset_by_name,
    load_tokenizer
)
from transformers import GPT2LMHeadModel, GPT2Config
import json
from utils.inference import inference
import os
import torch
from tqdm import tqdm
import pickle
import re
import random


def mask_sentences(dataset, mask_sents_num: str):
    result = []
    for data in tqdm(dataset):
        sentence_spliter = re.compile(r'([，,。：,:；;！!？?])')
        sentences = sentence_spliter.split(data['article'])
        sentences = list(filter(lambda s: s != '', sentences))
        masked_sent_num = 0
        while True:
            if len(list(filter(lambda s: s not in "，,。：,:；;！!？?", sentences))) < mask_sents_num:
                break
            choose_id = random.choice(range(len(sentences)))
            if not re.match(r'[，,。：,:；;！!？?]', sentences[choose_id]) and sentences[choose_id] != '[MASK_S]':
                masked_sent_num += 1
                sentences[choose_id] = '[MASK_S]'
            if masked_sent_num >= mask_sents_num:
                break
        data['masked_article'] = ''.join(sentences)
        del data['answer']
        result.append(data)
    return result


def create_MN_data(
    ckpt_path: str,
    dataset_name: str,
    tokenizer_name: str,
    max_seq_len: int,
    p: float,
    data_num: int,
    use_test_data: bool = True,
    mask_sents_num: int = None,
):
    # Load dataset.
    dataset = load_dataset_by_name(
        dataset_name=dataset_name
    )

    # Select specify dataset.
    if use_test_data:
        if mask_sents_num:
            dataset = mask_sentences(
                dataset=dataset['test'], mask_sents_num=mask_sents_num)
        data_iter = tqdm(dataset)
    else:
        if mask_sents_num:
            dataset = mask_sentences(
                dataset=dataset['train'], mask_sents_num=mask_sents_num)
        data_iter = tqdm(dataset)

    # Store data in `MN_dataset`.
    MN_dataset = []
    for data in data_iter:
        # Check if data amount more than `data_num`.
        # If so than stop generate data.
        # Just treat remain data as positive data.
        if len(MN_dataset) >= data_num:
            MN_dataset.append({
                'id': data['id'],
                'article': data['article'],
                'label': 1,
                'origin_article': data['article'],
            })
            continue

        try:
            # Get inference result.
            infr_result = inference(
                ckpt_path=ckpt_path,
                tokenizer_name=tokenizer_name,
                max_seq_len=max_seq_len,
                prompt='[ARTICLE]' + data['masked_article'] + '[SEP]',
                p=p
            )
            # Remove whitespace between tokens.
            infr_result = infr_result.replace(' ', '')
        except:
            continue
        # If inference result format error then continue.
        if not infr_result.endswith('[END]'):
            continue
        if not infr_result.startswith('[ARTICLE]'):
            continue

        # Remove bos and eos in inference result.
        infr_result = infr_result.replace('[ARTICLE]', '')
        infr_result = infr_result.replace('[END]', '')

        # Parse inference result to article.
        # (Use generated tokens to infill `[MASK]` token).
        try:
            if infr_result.find('[MASK]') != -1:
                # MLM dataset version littler than `MLM_dataset_v3`.
                answers = infr_result.split('[SEP]')[1].split('[MASK]')
                article = infr_result.split('[SEP]')[0]
                for ans in answers:
                    if ans == '':
                        raise Exception('Empty answer.')
                    article = article.replace('[MASK]', ans, 1)
            elif infr_result.find('[ANS]') != -1:
                # MLM dataset version above than `MLM_dataset_v3`.
                answers = infr_result.split('[SEP]')[1].split('[ANS]')[:-1]
                article = infr_result.split('[SEP]')[0]
                for ans in answers:
                    if ans == '':
                        raise Exception('Empty answer.')
                    article = re.sub(r'\[MASK_.\]', ans, article, 1)
            MN_dataset.append({
                'id': data['id'],
                'article': article,
                'label': 0,
                'origin_article': data['article'],
            })
        except Exception as err:
            # If inference result has error then treat it as positive data.
            MN_dataset.append({
                'id': data['id'],
                'article': data['article'],
                'label': 1,
                'origin_article': data['article'],
            })
            pass

    return MN_dataset


if __name__ == '__main__':
    MN_dataset = create_MN_data(
        ckpt_path='checkpoint/MLM_exp9/checkpoint-2724221.pt',
        dataset_name='MLM_dataset_v3',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        p=0.9,
        data_num=50000,
        use_test_data=False,
        mask_sents_num=3
    )
    pickle.dump(MN_dataset, open('MN_dataset_sent3.pk', 'wb'))
