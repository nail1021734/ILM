import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from _path import _PATH


def load_data():
    return load_dataset(
        os.path.join(_PATH, 'Taiwan_news_dataset.py'))


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
        'eos_token': '[END]'
    })
    tokenizer.model_max_length = max_length

    return tokenizer


def create_data_loader(batch_size, tokenizer_name, max_length):
    # Load dataset.
    datasets = load_data()
    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=tokenizer_name,
        max_length=max_length
    )

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

    # Create data loader.
    data_loader = DataLoader(
        dataset=datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=tokenizer_function
    )

    return data_loader


if __name__ == '__main__':
    data_loader = create_data_loader(2, 'chinese_tokenizer', max_length=128)
