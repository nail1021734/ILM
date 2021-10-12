import os
from typing import List

from datasets import load_dataset
from more_itertools import chunked
from tokenizers import (Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, trainers)


def build_BPE_tokenizer(tokenizer_name: str, dataset: List[str], vocab_size: int):
    # Set tokenizer parameter.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoders = decoders.ByteLevel()

    # Set special tokens.
    special_tokens_list = ["[PAD]", "<unk>", "[CLS]", "[SEP]", "[MASK]",
    "[TITLE]", "[ARTICLE]", "[REF]", "[END]", "<en>", "<num>",
    "[COMPANY]", "[REPORTER]", "[DATETIME]", "[CATEGORY]"]
    special_tokens_list.extend([f'<per{i}>' for i in range(30)])
    special_tokens_list.extend([f'<org{i}>' for i in range(30)])
    special_tokens_list.extend([f'<loc{i}>' for i in range(30)])

    # Create batches of data.
    corpus = chunked((
            f'[COMPANY]{i["company"]}' +
            f'[REPORTER]{i["reporter"]}'+
            f'[DATETIME]{i["datetime"]}'+
            f'[CATEGORY]{i["category"]}'+
            f'[TITLE]{i["title"]}'+
            f'[ARTICLE]{i["article"]}[END]' for i in dataset['train']), 1000)

    # Set trainer.
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens_list
    )

    # Training tokenizer.
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # # Set post_precessor.
    # tokenizer.post_processor = TemplateProcessing(
    #     single="[TITLE] $A [END]",
    #     pair="[TITLE] $A [ARTICLE] $B [END]",
    #     special_tokens=[
    #         ("[TITLE]", tokenizer.token_to_id('[TITLE]')),
    #         ("[ARTICLE]", tokenizer.token_to_id('[ARTICLE]')),
    #         ("[END]", tokenizer.token_to_id('[END]'))
    #     ]
    # )

    # Create directory to save tokenizer
    if not os.path.exists('tokenizer'):
        os.makedirs('tokenizer')

    # Save in directory.
    tokenizer.save(os.path.join('tokenizer', f'{tokenizer_name}.json'))


if __name__ == '__main__':
    dataset = load_dataset('Taiwan_news_dataset.py')
    build_BPE_tokenizer(
        tokenizer_name='chinese_tokenizer_big',
        dataset=dataset,
        vocab_size=50000,
    )
