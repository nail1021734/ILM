import os
import sqlite3
from datetime import datetime as dt

import datasets

from _path import ROOT_PATH

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Collect Taiwan news from about 10 news media.
"""

_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'ettoday_train.db')
_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'ettoday_test.db')

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


class TaiwanNewsConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        r"""BuilderConfig for TaiwanNewsDataset.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TaiwanNewsConfig, self).__init__(**kwargs)


class TaiwanNewsDataset(datasets.GeneratorBasedBuilder):
    r"""Collect Taiwan news from about 10 news media."""
    BUILDER_CONFIG = [
        TaiwanNewsConfig(
            name='news_text',
            version=datasets.Version('1.0.0', ''),
            description='news text'
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
        # Connect to DB.
        conn = sqlite3.connect(filepath)

        # Set `conn.row_factory` to get right return format.
        # conn.row_factory = lambda cursor, row: row[0]

        # Get database cursor.
        cursor = conn.cursor()

        sql = """
        SELECT id, datetime, company_id, category, reporter, title, article from news;
        """
        key = 0
        for (
            idx, datetime, company_id, category, reporter, title, article
        ) in cursor.execute(sql):
            if not (datetime and category and reporter and title and article):
                continue
            yield (
                key,
                {
                    'id': idx,
                    'datetime': dt.fromtimestamp(datetime).strftime('%Y-%d-%m'),
                    'company': COMPANY_DICT[company_id],
                    'category': category,
                    'reporter': reporter,
                    'title': title,
                    'article': article,
                }
            )
            key += 1
