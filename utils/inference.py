import json
import os
import re
import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from _path import ROOT_PATH
from utils.data_processor import load_tokenizer


def top_p(
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_seq_len: int,
    p: float,
):
    device = next(model.parameters()).device

    prev_tkids = tokenizer(prompt, return_tensors='pt')

    # Move tensors to model running device.
    prev_tkids = prev_tkids.to(device)

    # Get input ids.
    prev_tkids = prev_tkids.input_ids

    # Calculate how many token can be generate at most.
    out_seq_len = max_seq_len - prev_tkids.shape[1]
    if out_seq_len < 0:
        raise Exception('`prompt length` > `max_seq_length`')

    # Generate tokens.
    for _ in range(out_seq_len):
        next_tkids_probs = torch.nn.functional.softmax(
            model(input_ids=prev_tkids).logits,
            dim=-1
        )

        next_tkid_probs = next_tkids_probs[:, -1]

        (topk_tkid_probs, topk_tkid, ) = \
            next_tkid_probs.sort(dim=-1, descending=True)

        k = (topk_tkid_probs.cumsum(dim=-1) < p).sum().item()

        if k == 0:
            k = 1

        topk_tkid_probs = topk_tkid_probs[..., :k]
        topk_tkid = topk_tkid[..., :k]

        next_tkid_cand_idx = torch.multinomial(
            topk_tkid_probs,
            num_samples=1,
        )
        next_tkid = torch.gather(
            topk_tkid,
            -1,
            next_tkid_cand_idx,
        )

        prev_tkids = torch.cat(
            [prev_tkids, next_tkid],
            dim=-1
        )

        # If the prediction token id is `[END]`, then stop prediction.
        if next_tkid[0, 0].item() == tokenizer.eos_token_id:
            break

    # Output generated text.
    return tokenizer.decode(
        token_ids=prev_tkids[0],
    )


def inference(
    ckpt_path: str,
    tokenizer_name: str,
    max_seq_len: int,
    prompt: str,
    p: float,
):
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

    return top_p(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_seq_len=max_seq_len,
        p=p
    )


def format(infr_result: str):
    if not infr_result.endswith('[END]'):
        raise Exception('Input format error.')
    if not infr_result.startswith('[ARTICLE]'):
        raise Exception('Input format error.')
    infr_result = infr_result.replace('[ARTICLE]', '')
    if infr_result.find('[MASK]') != -1:
        # MLM dataset version littler than `MLM_dataset_v3`.
        answers = infr_result.split('[SEP]')[1].split('[MASK]')
        article = infr_result.split('[SEP]')[0]
        for ans in answers:
            article = article.replace('[MASK]', f'=={ans}==', 1)
    elif infr_result.find('[ANS]') != -1:
        # MLM dataset version above than `MLM_dataset_v3`.
        answers = infr_result.split('[SEP]')[1].split('[ANS]')
        article = infr_result.split('[SEP]')[0]
        for ans in answers:
            article = re.sub(r'\[MASK_.*?\]', f'=={ans}==', article, 1)
    else:
        raise Exception('Input error')
    article = ''.join(article.split(' '))
    return article
