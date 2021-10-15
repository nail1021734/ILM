import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
import torch
from _path import _PATH


def load_GPT_data():
    return load_dataset(
        os.path.join(_PATH, 'Taiwan_news_dataset.py'))

def load_MLM_data():
    return load_dataset(
        os.path.join(_PATH, 'MLM_dataset.py'))

def load_tokenizer(tokenizer_name, max_length):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(
        _PATH,
        'tokenizer',
        f'{tokenizer_name}.json'
    ))
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '[TITLE]',
        'sep_token': '[ARTICLE]',
        'eos_token': '[END]',
        'mask_token': '[MASK]',
        'unk_token': '<unk>',
    })
    tokenizer.model_max_length = max_length

    return tokenizer

def callate_fn_creater(tokenizer, training_mode):
    if training_mode == 'LM':
        # Get tokenized dataset.
        def tokenizer_function(data):
            sample = [
                f'[COMPANY]{i["company"]}' +
                f'[REPORTER]{i["reporter"]}'+
                f'[DATETIME]{i["datetime"]}'+
                f'[CATEGORY]{i["category"]}'+
                f'[TITLE]{i["title"]}'+
                f'[ARTICLE]{i["article"]}[END]' for i in data]
            tokenized_result = tokenizer(
                sample, truncation=True, padding=True, return_tensors='pt')
            del tokenized_result['token_type_ids']

            return tokenized_result
        return tokenizer_function
    elif training_mode == 'MLM':
        # Get tokenized dataset.
        def tokenizer_function(data):
            sample = [
                f'[ARTICLE]{i["masked_article"]}[SEP]{i["answer"]}[END]' for i in data]
            tokenized_result = tokenizer(
                sample, truncation=True, padding=True, return_tensors='pt')

            sep_id = tokenizer.get_vocab()['[SEP]']
            input_tensor = tokenized_result['input_ids']
            create_ans_mask = torch.arange(input_tensor.shape[-1])
            create_ans_mask = create_ans_mask > \
                (input_tensor == sep_id).nonzero()[:, -1].unsqueeze(dim=-1)
            tokenized_result['answer_mask'] = create_ans_mask
            del tokenized_result['token_type_ids']

            return tokenized_result
        return tokenizer_function

def create_data_loader(batch_size, tokenizer_name, max_length, training_mode='GPT'):
    # Load dataset.
    if training_mode == 'GPT':
        datasets = load_GPT_data()
    elif training_mode == 'MLM':
        datasets = load_MLM_data()

    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

    # Create data loader.
    data_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=callate_fn_creater(
            tokenizer=tokenizer,
            training_mode=training_mode
        )
    )

    return data_loader


if __name__ == '__main__':
    data_loader = create_data_loader(2, 'chinese_tokenizer', max_length=128)
