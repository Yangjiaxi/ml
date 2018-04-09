import numpy as np
import os
from collections import Counter
import re


# train_pos_path = 'aclImdb/train/pos/'
# train_neg_path = 'aclImdb/train/neg/'
# test_pos_path = 'aclImdb/test/pos/'
# test_neg_path = 'aclImdb/test/neg/'


def load_stop_words(file):
    swl = []
    with open(file, encoding='utf-8') as sw:
        swl.extend(sw.read().split('\n'))
    return np.array(swl)


def read_folder(path, stop_words):
    for files in (os.path.join(path, files_name) for files_name in os.listdir(path)):
        with open(files, encoding='utf-8') as txt:
            uni = np.array(list(set(re.findall(r'\w+', re.sub('<br />', ' ', txt.read().lower())))))
            yield np.setdiff1d(uni, stop_words)


def get_word_frequency(path, stop_words):
    files_count = len(os.listdir(path))
    i = 1
    total = Counter()
    for t in read_folder(path, stop_words):
        each = Counter(t)
        total.update(each)
        if i % (0.1 * files_count) == 0:
            print(" %d/%d " % (i, files_count), end='..')
        i += 1
    print("\nSuccess!")
    res = np.array(total.most_common())
    res[:, 1] = res[:, 1].astype('int32') / files_count
    res[:, 1] = np.log(res[:, 1].astype('float64'))  # 取对数，PI(a,b,c,d) = SIGMA(a,b,c,d) 过小的数字乘起来会趋0
    return res


def prefix(path, stop_words):
    with open(path, encoding='utf-8') as txt:
        uni = np.array(list(set(re.findall(r'\w+', re.sub('<br />', ' ', txt.read().lower())))))
        return np.setdiff1d(uni, stop_words)


def load_test(path, goal, stop_words, train_pos_dict, train_neg_dict):
    files_count = len(os.listdir(path))
    i = 1
    err_count = 0
    for files in (os.path.join(path, files_name) for files_name in os.listdir(path)):
        if i % (0.1 * files_count) == 0:
            print("after %d epochs, acc is %f" % (i, (1 - float(err_count) / files_count) * 100))
        fixed = prefix(files, stop_words)
        pos_score = 0.0
        neg_score = 0.0
        for word in fixed:
            if word in train_pos_dict:
                pos_score += float(train_pos_dict[word])
            if word in train_neg_dict:
                neg_score += float(train_neg_dict[word])
        i += 1
        if (pos_score - neg_score) * goal >= 0:
            continue
        else:
            err_count += 1
    return (1 - float(err_count) / files_count) * 100


def model(train_pos_path, train_neg_path, test_pos_path, test_neg_path, stop_words_path):
    d = {'train_pos_path': train_pos_path,
         'train_neg_path': train_neg_path,
         'test_pos_path': test_pos_path,
         'test_neg_path': test_neg_path}
    stop_words = load_stop_words(stop_words_path)
    print("*******Loading Positive Data*************************")
    train_pos_data = get_word_frequency(train_pos_path, stop_words)
    print("*******Loading Negative Data*************************")
    train_neg_data = get_word_frequency(train_neg_path, stop_words)
    train_pos_dict = dict(train_pos_data)
    train_neg_dict = dict(train_neg_data)
    print("*******Positive Tests:*******************************")
    d['test_pos_acc'] = load_test(test_pos_path, 1, stop_words, train_pos_dict, train_neg_dict)
    print("*******Negative Tests:*******************************")
    d['test_neg_acc'] = load_test(test_neg_path, -1, stop_words, train_pos_dict, train_neg_dict)
    return d


if __name__ == '__main__':
    result = model('aclImdb/train/pos/',
                   'aclImdb/train/neg/',
                   'aclImdb/test/pos/',
                   'aclImdb/test/neg/',
                   'stop_words.txt')
    for k in result:
        print(str(k) + " : " + str(result[k]))
