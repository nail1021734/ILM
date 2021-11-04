import os
import pickle
import sqlite3
from datetime import datetime as dt

import datasets

from _path import ROOT_PATH

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Collect Taiwan news from about 10 news media.
不將NER出來某類別的名詞使用tag替換掉
"""

_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'ettoday_train_notag.pk')
_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'ettoday_test_notag.pk')

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


class TaiwanNewsNTConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        r"""BuilderConfig for TaiwanNewsDataset.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TaiwanNewsNTConfig, self).__init__(**kwargs)


class TaiwanNewsNTDataset(datasets.GeneratorBasedBuilder):
    r"""Collect Taiwan news from about 10 news media."""
    BUILDER_CONFIG = [
        TaiwanNewsNTConfig(
            name='TaiwanNewsNTDataset',
            version=datasets.Version('1.0.0', ''),
            description='TaiwanNewsNTDataset'
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
                    'datetime': dt.fromtimestamp(data_dict['datetime']).strftime('%Y-%d-%m'),
                    'company': COMPANY_DICT[data_dict['company_id']],
                    'category': data_dict['category'],
                    'reporter': data_dict['reporter'],
                    'title': data_dict['title'],
                    'article': data_dict['article'],
                }
            )
            key += 1
