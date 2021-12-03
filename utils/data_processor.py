import os

import torch
from datasets import load_dataset
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from _path import ROOT_PATH


def load_LM_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'Taiwan_news_dataset.py'))


def load_MLM_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset.py'))


def load_MLM_data_v2():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v2.py'))


def load_MLM_data_v3():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v3.py'))


def load_MLM_data_v4():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v4.py'))


def load_MLM_data_v5():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_v5.py'))


def load_MLM_NT_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'MLM_dataset_NT.py'))


def load_LM_NT_data():
    return load_dataset(
        os.path.join(ROOT_PATH, 'dataset_script', 'Taiwan_news_dataset_notag.py'))


def load_tokenizer(tokenizer_name, max_length):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(
        ROOT_PATH,
        'tokenizer',
        f'{tokenizer_name}.json'
    ))
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'bos_token': '[TITLE]',
        'sep_token': '[ARTICLE]',
        'eos_token': '[END]',
        'mask_token': '[MASK_W]',
        'unk_token': '<unk>',
        'additional_special_tokens': [
            '[ANS]',
            '[MASK_S]',
            '[MASK_D]',
            '[MASK_N]',
        ]
    })
    tokenizer.model_max_length = max_length

    return tokenizer


def callate_fn_creater(tokenizer):
    # Set post processor to add special token.
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        pair="[ARTICLE] $A [SEP] $B [END]",
        special_tokens=[
            ("[ARTICLE]", tokenizer.get_vocab()["[ARTICLE]"]),
            ("[SEP]", tokenizer.get_vocab()["[SEP]"]),
            ("[END]", tokenizer.get_vocab()["[END]"]),
        ]
    )

    def tokenizer_function(data):
        masked_articles = [i["masked_article"] for i in data]
        answers = [i["answer"] for i in data]
        tokenized_result = tokenizer(
            masked_articles, answers, truncation=True, padding=True, return_tensors='pt')

        sep_id = tokenizer.get_vocab()['[SEP]']
        input_tensor = tokenized_result['input_ids']
        create_ans_mask = torch.arange(input_tensor.shape[-1])
        create_ans_mask = create_ans_mask > \
            (input_tensor == sep_id).nonzero()[:, -1].unsqueeze(dim=-1)
        tokenized_result['answer_mask'] = create_ans_mask
        del tokenized_result['token_type_ids']

        return tokenized_result
    return tokenizer_function


def load_dataset_by_name(dataset_name: str):
    if dataset_name == 'Taiwan_news_dataset':
        return load_LM_data()
    elif dataset_name == 'MLM_dataset':
        return load_MLM_data()
    elif dataset_name == 'MLM_dataset_v2':
        return load_MLM_data_v2()
    elif dataset_name == 'MLM_dataset_v3':
        return load_MLM_data_v3()
    elif dataset_name == 'MLM_dataset_v4':
        return load_MLM_data_v4()
    elif dataset_name == 'MLM_dataset_v5':
        return load_MLM_data_v5()
    elif dataset_name == 'LM_NT_data':
        return load_LM_NT_data()
    elif dataset_name == 'MLM_NT_data':
        return load_MLM_NT_data()
    else:
        raise ValueError(f'Dataset name not exist {dataset_name}')


def create_data_loader(
    batch_size: int,
    tokenizer_name: str,
    max_length: int,
    dataset_name: str,
    shuffle: bool = True,
    testing: bool = False,
):
    # Load dataset.
    datasets = load_dataset_by_name(dataset_name=dataset_name)

    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )
    if testing:
        dataset = datasets['test']
    else:
        dataset = datasets['train']
    # Create data loader.
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=callate_fn_creater(
            tokenizer=tokenizer,
        )
    )

    return data_loader


if __name__ == '__main__':
    data_loader = create_data_loader(2, 'chinese_tokenizer', max_length=128)
