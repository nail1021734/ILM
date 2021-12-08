import os
import pickle

import datasets

from _path import ROOT_PATH

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Collect Taiwan news from about 10 news media.
由實驗結果可以推測, 生成時有 format error 最大的問題是訓練時的單一mask數量差別太大，
雖然 mask 的總 token 數量差不多, 但是單一一種 mask 的數量差太多, 此資料集為將20%的
token mask 起來並且只做 word level 的 mask, 可以用這個資料集 finetune 訓練完的ILM.

產生資料時使用的設定如下:
- `document_mask_p`: 0.00
- `sentence_mask_p`: 0.0
- `word_mask_p`: 0.20
- `ngram_mask_p`: 0.0
- `min_ngram_length`: 2
- `max_ngram_length`: 6

大約mask總文章量的20%
"""

_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'mlm_train_token.pk')
_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'mlm_test_token.pk')

COMPANY_DICT = {
    1: 'chinatimes',
    2: 'cna',
    3: 'epochtimes',
    4: 'ettoday',
    5: 'ftv',
    6: 'ltn',
    7: 'ntdtv',
    8: 'setn',
    9: 'storm',
    10: 'tvbs',
    11: 'udn',
}


class MLMConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        r"""BuilderConfig for TaiwanNewsDataset.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(MLMConfig, self).__init__(**kwargs)


class MLMDataset(datasets.GeneratorBasedBuilder):
    r"""Collect Taiwan news from about 10 news media."""
    BUILDER_CONFIG = [
        MLMConfig(
            name='mlm_news_token',
            version=datasets.Version('1.0.0', ''),
            description='mlm news token'
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    'id': datasets.Value('int32'),
                    'title': datasets.Value('string'),
                    'article': datasets.Value('string'),
                    'reporter': datasets.Value('string'),
                    'datetime': datasets.Value('string'),
                    'category': datasets.Value('string'),
                    'company': datasets.Value('string'),
                    'masked_article': datasets.Value('string'),
                    'answer': datasets.Value('string'),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": _TRAIN_PATH}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": _TEST_PATH}),
        ]

    def _generate_examples(self, filepath):
        dataset = pickle.load(open(filepath, 'rb'))
        key = 0
        for data_dict in dataset:
            if not (data_dict['datetime'] and
                    data_dict['category'] and
                    data_dict['reporter'] and
                    data_dict['title'] and
                    data_dict['article']
                    ):
                continue
            yield (
                key,
                {
                    'id': data_dict['id'],
                    'datetime': data_dict['datetime'],
                    'company': data_dict['company'],
                    'category': data_dict['category'],
                    'reporter': data_dict['reporter'],
                    'title': data_dict['title'],
                    'article': data_dict['article'],
                    'masked_article': data_dict['masked_article'],
                    'answer': data_dict['answer'],
                }
            )
            key += 1
