import os
import sqlite3
from datetime import datetime as dt

import datasets

from _path import ROOT_PATH

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """
Collect Taiwan news from about storm, ettoday, and cna.
"""

_TRAIN_PATH = os.path.join(ROOT_PATH, 'dataset', 'merge_train.db')
_TEST_PATH = os.path.join(ROOT_PATH, 'dataset', 'merge_test.db')

COMPANY_DICT = {
    0: 'chinatimes',
    1: 'cna',
    2: 'epochtimes',
    3: 'ettoday',
    4: 'ftv',
    5: 'ltn',
    6: 'ntdtv',
    7: 'setn',
    8: 'storm',
    9: 'tvbs',
    10: 'udn',
}


class TaiwanNewsConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        r"""BuilderConfig for TaiwanNewsDataset.

        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super(TaiwanNewsConfig, self).__init__(**kwargs)


class TaiwanNewsDataset(datasets.GeneratorBasedBuilder):
    r"""Collect Taiwan news from about storm, ettoday, and cna."""
    BUILDER_CONFIG = [
        TaiwanNewsConfig(name='news_text',
                         version=datasets.Version('1.0.0', ''),
                         description='news text'),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                'id': datasets.Value('int32'),
                'title': datasets.Value('string'),
                'article': datasets.Value('string'),
                'reporter': datasets.Value('string'),
                'timestamp': datasets.Value('string'),
                'category': datasets.Value('string'),
                'company': datasets.Value('string'),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": _TRAIN_PATH}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": _TEST_PATH}),
        ]

    def _generate_examples(self, filepath):
        # Connect to DB.
        conn = sqlite3.connect(filepath)

        # Set `conn.row_factory` to get right return format.
        # conn.row_factory = lambda cursor, row: row[0]

        # Get database cursor.
        cursor = conn.cursor()

        sql = """
        SELECT id, timestamp, company_id, category, reporter, title, article from parsed_news;
        """
        key = 0
        for (idx, timestamp, company_id, category, reporter, title,
             article) in cursor.execute(sql):
            if not (timestamp and category and reporter and title and article):
                continue
            yield (key, {
                'id':
                idx,
                'timestamp':
                dt.fromtimestamp(timestamp).strftime('%Y-%d-%m'),
                'company':
                COMPANY_DICT[company_id],
                'category':
                category,
                'reporter':
                reporter,
                'title':
                title,
                'article':
                article,
            })
            key += 1
