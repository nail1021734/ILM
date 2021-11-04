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


if __name__ == '__main__':
    sen = inference(
        ckpt_path='checkpoint/MLM_exp8_weight_Decay_error/checkpoint-2700000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompt='[ARTICLE]「第<num>任總統副總統及第<num>屆立法委員選舉」<num>年<num>月<num>日投票,<loc0>選舉委員會總幹事<per0><num>日表示,[MASK_S],首投族有<num>萬多人,投開票所也增加<num>所,[MASK_S]。 <per0>說,這次選舉分為總統、立委及政黨票<num>種,每個投票所將設置<num>個選舉票匭,其中,[MASK_S],<loc0>登記參選立委有<num>人、政黨票有<num>組,不論是立委參選人數及政黨組都創歷年來新高。 <loc1>補充說明,<loc2>人口數已達<num>萬<num>人,比去年同期增加<num>萬<num>人,其中北屯區人口數更是大幅提升近<num>千人,此外,今年首投族人數統計至<num>月<num>日止有為<num>萬餘人,因此投開票所數量也從<num>所擴增至<num>所,以確保投票更為順暢快速。 <org0>表示,已請各區公所及<loc0>各機關利用網站、<en>群組鼓勵民眾踴躍投票,[MASK_S],攜帶本人身分證、印章及投票通知單,依投票通知單所載投票所地點前往投票。[SEP]',
        p=0.9
    )
    print(format(sen))
