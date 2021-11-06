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
from typing import Dict


def mask_sentences(dataset, mask_range: Dict, max_fail_count: int):
    result = []
    for data in tqdm(dataset):
        sentence_spliter = re.compile(r'([，,。：,:；;！!？?])')
        sentences = sentence_spliter.split(data['article'])
        masked_rate = 0
        fail_count = 0
        while True:
            if mask_range['max'] > masked_rate > mask_range['min']:
                break
            choose_id = random.choice(range(len(sentences)))
            if (re.match(r'[^，,。：,:；;！!？?]', sentences[choose_id]) and
                    sentences[choose_id] != '[MASK_S]'):
                sent_mask_rate = len(
                    sentences[choose_id]) / len(data['article'])
                if masked_rate + sent_mask_rate > mask_range['max']:
                    fail_count += 1
                    if fail_count >= max_fail_count:
                        break
                    continue
                sentences[choose_id] = '[MASK_S]'
                masked_rate += sent_mask_rate
        if max_fail_count >= fail_count:
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
    mask_range: Dict = None,
    mask_fail_count: int = 15,
):
    # Load dataset.
    dataset = load_dataset_by_name(
        dataset_name=dataset_name
    )

    # Select specify dataset.
    if use_test_data:
        if mask_range:
            dataset = mask_sentences(
                dataset=dataset['test'],
                mask_range=mask_range,
                max_fail_count=mask_fail_count
            )
        data_iter = tqdm(dataset)
    else:
        if mask_range:
            dataset = mask_sentences(
                dataset=dataset['train'],
                mask_range=mask_range,
                max_fail_count=mask_fail_count
            )
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
                'title': data['title'],
                'reporter': data['reporter'],
                'datetime': data['datetime'],
                'category': data['category'],
                'company': data['company'],
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
                'title': data['title'],
                'reporter': data['reporter'],
                'datetime': data['datetime'],
                'category': data['category'],
                'company': data['company'],
            })
        except Exception as err:
            # If inference result has error then treat it as positive data.
            MN_dataset.append({
                'id': data['id'],
                'article': data['article'],
                'label': 1,
                'origin_article': data['article'],
                'title': data['title'],
                'reporter': data['reporter'],
                'datetime': data['datetime'],
                'category': data['category'],
                'company': data['company'],
            })
            pass

    return MN_dataset


if __name__ == '__main__':
    MN_dataset = create_MN_data(
        ckpt_path='checkpoint/MLM_exp10/checkpoint-3200000.pt',
        dataset_name='MLM_dataset_v3',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        p=0.9,
        data_num=50000,
        use_test_data=False,
        mask_range={'max': 0.35, 'min': 0.25},
    )
    pickle.dump(MN_dataset, open('MN_dataset_25~30%.pk', 'wb'))
