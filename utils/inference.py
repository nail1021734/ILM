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
        ckpt_path='checkpoint/MLM_exp10/checkpoint-3200000.pt',
        tokenizer_name='chinese_tokenizer_big',
        max_seq_len=512,
        prompt='[ARTICLE]<per0>育有<num>女<num>子,除了大女兒<per1>,小兒子<per2>也有在娛樂圈活動,[MASK_S],他哭笑不得地表示兒子有次無心的舉動,[MASK_S],「是不是幫我掛號讓我去問個醫生或怎麼樣?」後來才知道是烏龍一場。 <per0>最近在<unk>透露,<per2>曾讓房間內的電扇持續吹一整年,[MASK_S],一度想掛號就醫,一出門錄影卻又突然「好轉」,直到有次靠近兒子的房間時,耳鳴突然變得超大聲,他才終於抓到兇手! 往鋼琴底下一看,<per0>發現那裡有台電風扇,「連續吹風<num>年啊!」<per1>在旁邊有感而發地說:「這種男生要嘛只會關燈,[MASK_S],他沒辦法同時把這兩個地方都關了再出門!」 對此,<per2>事後解釋是因為擔心鋼琴會受潮,<per0>哭笑不得地說:「我終於在<per2>從<loc0>回<loc1>的時候,治好了我的耳鳴!恁爸差點就把他...!」對兒子是好氣又好笑。[SEP]',
        p=0.9
    )
    print(format(sen))
