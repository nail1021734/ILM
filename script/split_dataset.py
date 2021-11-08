import pickle
import sqlite3

if __name__ == '__main__':
    conn = sqlite3.connect('dataset/ettoday_merge.db')
    cursor = conn.cursor()
    total_data = list(cursor.execute('SELECT * from news;'))
    total_data = [
        {
            'id': i[0],
            'article': i[1],
            'category': i[2],
            'company_id': i[3],
            'datetime': i[4],
            'reporter': i[5],
            'title': i[6],
            'url_pattern': i[7]
        } for i in total_data
    ]
    test_size = 20000
    train_size = len(total_data) - test_size

    train_data = total_data[:train_size]
    test_data = total_data[train_size:]

    pickle.dump(train_data, open('dataset/ettoday_train_notag.pk', 'wb'))
    pickle.dump(test_data, open('dataset/ettoday_test_notag.pk', 'wb'))
