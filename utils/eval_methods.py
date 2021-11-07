import json
import os
import math
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel

from utils.data_processor import create_data_loader, load_tokenizer


def perplexity(
    model: GPT2LMHeadModel,
    data_loader: str,
    pad_id: int,
):
    total_perplex = 0
    device = next(model.parameters()).device
    criterion = CrossEntropyLoss(ignore_index=pad_id)

    for batch_inputs in tqdm(data_loader):
        batch_inputs['input_ids'] = batch_inputs['input_ids'].to(device)
        batch_inputs['attention_mask'] = batch_inputs['attention_mask'].to(
            device)
        outputs = model(
            input_ids=batch_inputs['input_ids'],
            attention_mask=batch_inputs['attention_mask']
        )

        # Calaulate loss.
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        # Start from second word.
        mask = batch_inputs['answer_mask'][..., 1:].to(device)
        padding_id = torch.zeros_like(mask) + pad_id
        padding_id = padding_id * (mask == False)
        padding_id = padding_id.to(device)
        shift_labels = batch_inputs['input_ids'][..., 1:].contiguous() \
            * mask + padding_id

        # Flatten the tokens
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        total_perplex += math.exp(loss.item())
    return total_perplex / len(data_loader.dataset)


def eval(
    eval_method: str,
    tokenizer_name: str,
    ckpt_path: str,
    max_seq_len: int,
    eval_dataset: str,
    shuffle: bool = False
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

    # Create data loader.
    data_loader = create_data_loader(
        batch_size=1,
        tokenizer_name=tokenizer_name,
        max_length=max_seq_len,
        dataset_name=eval_dataset,
        shuffle=shuffle,
        testing=True
    )

    if eval_method == 'perplexity':
        eval_result = perplexity(
            model=model,
            data_loader=data_loader,
            pad_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError(f'Eval method `{eval_method}` not defined.')

    return eval_result
