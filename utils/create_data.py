import re
from copy import deepcopy
from functools import partial
from random import choice, random, randrange
from typing import Callable, Dict, List

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from utils.data_processor import load_dataset_by_name, load_tokenizer
from utils.inference import format_article, inference


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
        return (
            ''.join(masked_sentences),
            ans_end_token.join(answer) + ans_end_token
        )


def mask_sentences_accr_token(
    dataset,
    tokenizer: PreTrainedTokenizerFast,
    mask_range: Dict,
    max_fail_count: int,
):
    r"""Mask sentences according token rate.(依照 mask 掉的 token 比例，決定
    還要不要 MASK)
    """
    result = []
    for data in tqdm(dataset):
        sentence_spliter = re.compile(r'([，,。：,:；;！!？?])')
        sentences = sentence_spliter.split(data['article'])
        masked_rate = 0
        fail_count = 0
        article_len = len(tokenizer.tokenize(data['article']))
        sentences_len = [len(tokenizer.tokenize(i)) for i in sentences]
        while True:
            if mask_range['max'] > masked_rate > mask_range['min']:
                break
            choose_id = choice(range(len(sentences)))
            if (re.match(r'[^，,。：,:；;！!？?]', sentences[choose_id]) and
                    sentences[choose_id] != '[MASK_S]'):
                sent_mask_rate = sentences_len[choose_id] / article_len
                if masked_rate + sent_mask_rate > mask_range['max']:
                    fail_count += 1
                    if fail_count >= max_fail_count:
                        break
                    continue
                sentences[choose_id] = '[MASK_S]'
                masked_rate += sent_mask_rate
        if max_fail_count >= fail_count:
            data['masked_article'] = ''.join(
                ['[ARTICLE]'] + sentences + ['[SEP]'])
            del data['answer']
            result.append(data)
    return result


def mask_sent_sent_rate(
    dataset,
    tokenizer: PreTrainedTokenizerFast,
    mask_range: Dict,
    max_fail_count: int,
):
    r"""Mask sentences according sentence rate.(依照 mask 掉的句子比例，決定
    還要不要 MASK)
    """
    result = []
    for data in tqdm(dataset):
        sentence_spliter = re.compile(r'([，,。：,:；;！!？?])')
        sentences = sentence_spliter.split(data['article'])
        masked_rate = 0
        fail_count = 0
        article_len = len(tokenizer.tokenize(data['article']))
        sentences_len = [len(tokenizer.tokenize(i)) for i in sentences]
        while True:
            if mask_range['max'] > masked_rate > mask_range['min']:
                break
            choose_id = choice(range(len(sentences)))
            if (re.match(r'[^，,。：,:；;！!？?]', sentences[choose_id]) and
                    sentences[choose_id] != '[MASK_S]'):
                sent_mask_rate = sentences_len[choose_id] / article_len
                if masked_rate + sent_mask_rate > mask_range['max']:
                    fail_count += 1
                    if fail_count >= max_fail_count:
                        break
                    continue
                sentences[choose_id] = '[MASK_S]'
                masked_rate += sent_mask_rate
        if max_fail_count >= fail_count:
            data['masked_article'] = ''.join(
                ['[ARTICLE]'] + sentences + ['[SEP]'])
            del data['answer']
            result.append(data)
    return result


def re_generate_fail_data(
    fail_articles: List[str],
    infr_function: Callable,
    max_fail_times: int = 3,
):
    regenerated_articles = [''] * len(fail_articles)
    re_generated_id = list(range(len(fail_articles)))
    prompts = deepcopy(fail_articles)

    # Try generate failed article `max_fail_times` times.
    for _ in range(max_fail_times):
        # If all article generate success then break.
        if len(re_generated_id) == 0:
            break
        print(f'Regenerate {len(re_generated_id)} articles.')

        # Generate articles.
        infr_result = infr_function(prompts=prompts)

        # Reset `prompts` to store still fail articles for next iteration.
        prompts = []
        for index, article in zip(
            deepcopy(re_generated_id),
            infr_result
        ):
            try:
                # Format article to check whether article format correct.
                formated_article = format_article(article)
                # If correct then save this article in it's index.
                regenerated_articles[index] = formated_article
                # Remove this article id in `re_generated_id` list.
                re_generated_id.remove(index)
            except Exception as err:
                print(err.args)
                # If still not correct then add failed articles in `prompts`
                # for next iteration.
                prompts.append(fail_articles[index])

    # Print amount of fail article.
    if len(re_generated_id) > 0:
        print(f'{len(re_generated_id)} articles still failed'
              + f' after try {max_fail_times} times.')

    return regenerated_articles


def create_MN_data(
    ckpt_path: str,
    dataset_name: str,
    tokenizer_name: str,
    max_seq_len: int,
    data_num: int,
    mask_strategy: str,
    p: float = None,
    k: float = None,
    max_prompt_len: int = 400,
    use_test_data: bool = True,
    mask_range: Dict = None,
    mask_fail_count: int = 15,
    batch_size: int = 8,
):
    # Load dataset.
    dataset = load_dataset_by_name(
        dataset_name=dataset_name
    )

    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_seq_len
    )

    # Select specify dataset.
    if use_test_data:
        dataset = dataset['test']
    else:
        dataset = dataset['train']

    # Drop article that amount of token larger than `max_prompt_len - 2`.
    # (Include `[ARTICLE]` and `[SEP]` so minus 2)
    data = []
    for i in tqdm(dataset):
        if len(tokenizer.tokenize(i['article'])) < max_prompt_len - 2:
            data.append(i)
    dataset = data

    # Split human article and machine article.
    dataset = dataset[:data_num*2]
    human_data = dataset[:len(dataset)//2]
    machine_data = dataset[len(dataset)//2:]

    # Adjust format.
    for data in human_data:
        data['origin_article'] = data['article']
        data['label'] = 1

    # Mask machine data.
    if mask_strategy == 'Sentence_token_rate':
        machine_data = mask_sentences(
            dataset=machine_data,
            mask_range=mask_range,
            max_fail_count=mask_fail_count,
            tokenizer=tokenizer,
        )
    else:
        raise Exception('Mask strategy not exist.')

    # Create partial inference function.
    partial_infr_func = partial(
        inference,
        ckpt_path=ckpt_path,
        tokenizer_name=tokenizer_name,
        max_seq_len=max_seq_len,
        p=p,
        k=k,
        batch_size=batch_size
    )

    # Inference machine data.
    infr_result = partial_infr_func(
        prompts=[i['masked_article'] for i in machine_data],
    )

    # Align `infr_result` with machine dataset.
    for index, data in enumerate(machine_data):
        data['origin_article'] = data['article']
        data['article'] = infr_result[index]
        data['label'] = 0

    # Format machine generate article.
    fail_index = []
    for index, data in enumerate(machine_data):
        try:
            data['article'] = format_article(data['article'])
        except:
            # Maybe generated article format is broken.
            data['article'] = ''
            fail_index.append(index)

    # Try to regenerate fail articles.
    regenerate_result = re_generate_fail_data(
        fail_articles=[machine_data[i]['masked_article'] for i in fail_index],
        infr_function=partial_infr_func,
        max_fail_times=5,
    )

    # Fill regenerated article in `machine_data`.
    for i, regen_article in zip(fail_index, regenerate_result):
        machine_data[i]['article'] = regen_article

    # Remove bad generate article.
    machine_data = list(
        filter(lambda data: data['article'] != '', machine_data))

    # Let human data has same amount article with machine data.
    human_data = human_data[:len(machine_data)]

    return human_data + machine_data


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
