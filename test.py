import random

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from data_processor import load_tokenizer

if __name__ == '__main__':
    # Config.
    # exp_name = '4_linear'
    max_length = 128
    input_prompt = '[TITLE]中央疫情指揮中心[ARTICLE]中央疫情指揮中心今日發表聲明'
    topk = 5

    # Set random seed.
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load pretrained tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name='chinese_tokenizer',
        max_length=max_length
    )

    # Load pretrained model.
    Config = GPT2Config.from_dict({"vocab_size": 50000, "n_ctx": 512, "n_positions": 512, "n_embd": 768, "n_layer": 12, "n_head": 12, "n_inner": 2048, "activation_function": "gelu_new"})
    model = GPT2LMHeadModel(Config)
    model.load_state_dict(
        torch.load('ray/test/DEFAULT_00d169ce_4_accumulate_step=20,batch_size=4,epoch_num=3,lr=6.3528e-05,seed=12,warm_up_step_rate=0.038766_2021-10-11_21-27-38/checkpoint-30000.pt', map_location='cpu'))
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
        outputs = model(**inputs)
        distribution = torch.nn.functional.softmax(outputs[0], dim=-1)
        topk_prob = distribution.topk(k=topk, dim=-1)
        select_id = torch.multinomial(topk_prob.values[0, -1], 1).item()
        new_id = topk_prob.indices[0, -1, select_id].unsqueeze(dim=0)
        output_ids = torch.cat(
            (inputs['input_ids'], new_id.unsqueeze(dim=0)), dim=-1)
        input_prompt = tokenizer.decode(output_ids[0])
    print(input_prompt.strip())
