import json
import os
import re
from typing import List, Union

import torch
from more_itertools import chunked
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from utils.data_processor import load_tokenizer


def clear_samping_result(
    sampling_result: torch.Tensor,
    padding_id: int,
    eos_id: int,
):
    # Remove no `[END]` article.
    clear_articles = []
    for i in sampling_result.tolist():
        if eos_id in i:
            clear_articles.append(i)
        else:
            clear_articles.append([])

    # Remove pad token in output.
    clear_articles = list(
        map(
            lambda sub_list: list(
                filter(lambda element: element != padding_id, sub_list)),
            clear_articles
        )
    )

    # Remove reduntdent tokens.
    def remove_reduntdent(sub_list):
        if sub_list:
            return sub_list[: sub_list.index(eos_id)+1]
        else:
            return sub_list
    clear_articles = list(
        map(
            remove_reduntdent,
            clear_articles,
        )
    )
    return clear_articles


def sampling(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    prompts: Union[str, List[str]],
    max_seq_len: int,
    k: float = None,
    p: float = None,
):
    device = next(model.parameters()).device

    prev_tkids = tokenizer(prompts, padding=True, return_tensors='pt')

    # Move tensors to model running device.
    prev_tkids = prev_tkids.to(device)

    result = model.generate(
        input_ids=prev_tkids.input_ids,
        attention_mask=prev_tkids.attention_mask,
        top_k=k,
        top_p=p,
        max_length=max_seq_len,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    clear_articles = clear_samping_result(
        sampling_result=result,
        padding_id=tokenizer.pad_token_id,
        eos_id=tokenizer.get_vocab()['[END]'],
    )

    # Output generated text.
    return tokenizer.batch_decode(
        clear_articles,
    )


def inference(
    ckpt_path: str,
    tokenizer_name: str,
    max_seq_len: int,
    prompts: Union[str, List[str]],
    batch_size: int = None,
    k: float = None,
    p: float = None,
):
    if k is None and p is None:
        raise Exception('Must give `k` or `p` parameter.(Not both.)')
    if k is not None and p is not None:
        raise Exception('Must give `k` or `p` parameter.(Not both.)')

    # Select device to inference.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get checkpoint directory.
    ckpt_dir = os.path.dirname(ckpt_path)

    # Load model_config.
    model_config = json.load(
        open(os.path.join(ckpt_dir, 'model_config.json'), 'r'))
    model_config = GPT2Config.from_dict(model_config)

    # Load model.
    model = GPT2LMHeadModel(model_config).to(device)
    model.load_state_dict(torch.load(ckpt_path))

    # Load tokenizer.
    tokenizer = load_tokenizer(tokenizer_name, max_length=max_seq_len)

    if batch_size is None:
        # Only inference one article,
        return sampling(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_seq_len=max_seq_len,
            k=k,
            p=p,
        )
    else:
        # Inference articles.
        result = []
        for batch in tqdm(list(chunked(prompts, batch_size))):
            result.extend(sampling(
                model=model,
                tokenizer=tokenizer,
                prompts=batch,
                max_seq_len=max_seq_len,
                k=k,
                p=p,
            ))
        return result


def format_article(infr_result: str):
    infr_result = ''.join(infr_result.split(' '))
    # Check if article format correct.
    if not infr_result.startswith('[ARTICLE]'):
        raise Exception('Input format error.')
    if not infr_result.endswith('[END]'):
        raise Exception('Input format error.')
    # Remove BOS and EOS.
    infr_result = infr_result.replace('[ARTICLE]', '')
    infr_result = infr_result.replace('[END]', '')

    answers = infr_result.split('[SEP]')[1].split('[ANS]')[:-1]
    article = infr_result.split('[SEP]')[0]

    if len(answers) != len(re.findall(r'\[MASK_[WSD]\]', article)):
        raise Exception('Input error.')
    if '' in answers:
        raise Exception('Generated article have empty answer.')
    for ans in answers:
        article = re.sub(r'\[MASK_[WSD]\]', f'{ans}', article, 1)
    article = ''.join(article.split(' '))

    return article
