import re
from random import random, randrange
from typing import Dict

from tqdm import tqdm

from utils.data_processor import load_dataset_by_name, load_tokenizer
from utils.inference import inference


def mask_article(
    tokenizer,
    article: str,
    document_mask_p: float,
    sentence_mask_p: float,
    word_mask_p: float,
    ngram_mask_p: float,
    min_ngram_length: int,
    max_ngram_length: int,
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


def mask_dataset(
    dataset_name: str,
    tokenizer_name: str,
    max_length: int,
    document_mask_p: float = 0.03,
    sentence_mask_p: float = 0.045,
    word_mask_p: float = 0.05,
    ngram_mask_p: float = 0.5,
    min_ngram_length: int = 2,
    max_ngram_length: int = 6,
):
    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name,
        max_length=max_length
    )
    # Load dataset.
    dataset = load_dataset_by_name(dataset_name=dataset_name)
    # Mask training data.
    train_data_result = []
    for data in tqdm(dataset['train']):
        data['masked_article'], data['answer'] = mask_article(
            tokenizer=tokenizer,
            article=data['article'],
            document_mask_p=document_mask_p,
            sentence_mask_p=sentence_mask_p,
            word_mask_p=word_mask_p,
            ngram_mask_p=ngram_mask_p,
            min_ngram_length=min_ngram_length,
            max_ngram_length=max_ngram_length,
        )
        train_data_result.append(data)

    # Mask test data.
    test_data_result = []
    for data in tqdm(dataset['test']):
        data['masked_article'], data['answer'] = mask_article(
            tokenizer=tokenizer,
            article=data['article'],
            document_mask_p=document_mask_p,
            sentence_mask_p=sentence_mask_p,
            word_mask_p=word_mask_p,
            ngram_mask_p=ngram_mask_p,
            min_ngram_length=min_ngram_length,
            max_ngram_length=max_ngram_length,
        )
        test_data_result.append(data)
    return train_data_result, test_data_result
