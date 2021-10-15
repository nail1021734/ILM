import random

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import os
import json
from data_processor import load_tokenizer

def inference(
    exp_ckpt_path: str,
    ckpt_name: str,
    tokenizer_name: str,
    input_prompt: str,
    max_length: int = 512,
    topk: int = 2,
):
    # Config.
    cfg_path = os.path.join(exp_ckpt_path, 'model_config.json')
    cfg = json.load(open(cfg_path, 'r'))
    model_config = GPT2Config.from_dict(cfg)

    # Set random seed.
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load pretrained tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    # Load pretrained model.
    model = GPT2LMHeadModel(model_config).to(device)
    model.load_state_dict(
        torch.load(os.path.join(exp_ckpt_path, ckpt_name)))
    # model.mask_controller.load_state_dict(
    #     torch.load('checkpoint/Yelp_min4star5/checkpoint-90000.pt', map_location='cpu'))

    for _ in range(max_length):
        inputs = tokenizer(
            input_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        inputs['input_ids'] = inputs['input_ids'].to(device)
        inputs['attention_mask'] = inputs['attention_mask'].to(device)
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        distribution = torch.nn.functional.softmax(outputs[0], dim=-1)
        topk_prob = distribution.topk(k=topk, dim=-1)
        select_id = torch.multinomial(topk_prob.values[0, -1], 1).item()
        new_id = topk_prob.indices[0, -1, select_id].unsqueeze(dim=0)
        output_ids = torch.cat(
            (inputs['input_ids'], new_id.unsqueeze(dim=0)), dim=-1)
        if tokenizer.decode(output_ids[0]) == '[END]':
            break
        input_prompt = tokenizer.decode(output_ids[0])
    print(input_prompt.strip())

if __name__ == '__main__':
    inference(
        exp_ckpt_path='checkpoint/MLM_exp3',
        ckpt_name='checkpoint-410000.pt',
        tokenizer_name='chinese_tokenizer_big',
        input_prompt='[ARTICLE][MASK],<num>日在粉嶺裁判法院提堂,[MASK],期間不得保釋。於此之前,<loc0>警方<num>日將該男子的犯行由傷人案改列作企圖[MASK]。<unk>報導,[MASK],<per0>左胸部被刺中,他的兩位同事也受了傷,後被送往醫院治療,肇事者被當場制服。<loc0>中聯辦負責人<num>日發表聲明,凶徒近距離持刀刺向<per0>心臟部位,妄圖直取性命,動機清晰明確。這種直接針對候選人的極端暴力犯罪行為,完全突破了法律底線、人性底線,嚴重損害<org1>選舉環境的公平公正和安全,「我們堅決支持特區有關[MASK]凶徒。」<unk>報導,<loc2><org3>新聞發言人<per1>於<num>日表示,[MASK],不僅是危害他人生命安全的嚴重犯罪行為,也是赤裸裸的選舉暴力,「我們對此表示極大憤慨和強烈譴責,對<per0>議員及其同事表示深切慰問。」據悉,<per0><num>日在<loc1>湖翠路時,遭到一名戴鴨舌帽的藍衣男子拿力捅傷胸口藍衣男被捕後不斷大喊「<per0>你天收!」、「你個人渣!」港警表示,事件導致<num>人受傷,包括[MASK]、<per0>助裡和持刀嫌犯,詳細案情目前仍在進一步調查中。[SEP]',
    )
