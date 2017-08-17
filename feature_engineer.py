import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, canberra, minkowski, braycurtis
from nltk import word_tokenize


# Set parameters
stop_words = set(stopwords.words('english'))
common_start = ['why', 'what', 'how', "what's", 'do', 'does', 'is',
                'can', 'which', 'if', 'i', 'are', 'where', 'who']


def word_len(s):
    return len(str(s))


def common_words(row):
    set1 = set(row['q1_split'])
    set2 = set(row['q2_split'])
    return len(set1.intersection(set2))


def common_words_unit(row):
    set1 = set(row['question1'])
    set2 = set(row['question2'])
    return len(set1.intersection(set2))


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        q1words[word] = 1
    for word in row['q2_split']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def word_match_share_stops(row, stops=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        if word not in stops:
            q1words[word] = 1
    for word in row['q2_split']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def tfidf_word_match_share_stops(row, stops=None, weights=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        if word not in stops:
            q1words[word] = 1
    for word in row['q2_split']:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def tfidf_word_match_share(row, weights=None):
    q1words = {}
    q2words = {}
    for word in row['q1_split']:
        q1words[word] = 1
    for word in row['q2_split']:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def calculate_tfidf(qs):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit_transform(qs)
    idf = vectorizer.idf_
    dict_tfidf = dict(zip(vectorizer.get_feature_names(), idf))

    print('\nMost common words and weights:')
    print(sorted(dict_tfidf.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
    print('\nLeast common words and weights: ')
    print(sorted(dict_tfidf.items(), key=lambda x: x[1], reverse=True)[:10])
    # print(dict_tfidf.get('how', 0))
    return dict_tfidf


def jaccard(row):
    wic = set(row['q1_split']).intersection(set(row['q2_split']))
    uw = set(row['q1_split']).union(row['q2_split'])
    if len(uw) == 0:
        uw = [1]
    return len(wic) / len(uw)


def total_unique_words(row):
    return len(set(row['q1_split']).union(row['q2_split']))


def total_unq_words_stop(row, stops):
    return len([x for x in set(row['q1_split']).union(row['q2_split']) if x not in stops])


def word_count(divided_s):
    return len(divided_s)


def wc_diff(row):
    return abs(len(row['q1_split']) - len(row['q2_split']))


def wc_ratio(row):
    l1 = len(row['q1_split']) * 1.0
    l2 = len(row['q2_split'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique(row):
    return abs(len(set(row['q1_split'])) - len(set(row['q2_split'])))


def wc_ratio_unique(row):
    l1 = len(set(row['q1_split'])) * 1.0
    l2 = len(set(row['q2_split']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['q1_split']) if x not in stops]) - len(
        [x for x in set(row['q2_split']) if x not in stops]))


def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['q1_split']) if x not in stops]) * 1.0
    l2 = len([x for x in set(row['q2_split']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def same_start_word(row):
    if not row['q1_split'] or not row['q2_split']:
        return np.nan
    return int(row['q1_split'][0] == row['q2_split'][0])


def same_end_word(row):
    if not row['q1_split'] or not row['q2_split']:
        return np.nan
    return int(row['q1_split'][-1] == row['q2_split'][-1])


def word_len_char(divided_s):
    return len(''.join(divided_s))


def len_char_diff(row):
    return abs(len(''.join(row['q1_split'])) - len(''.join(row['q2_split'])))


def char_ratio(row):
    l1 = len(''.join(row['q1_split']))
    l2 = len(''.join(row['q2_split']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2


def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['q1_split']) if x not in stops])) - len(
        ''.join([x for x in set(row['q2_split']) if x not in stops])))


def num_capital(s):
    return sum(1 for c in s if c.isupper())


def num_ques_mark(s):
    return sum(1 for c in s if c is '?')


def start_with(divided_s, start):
    if divided_s:
        return 1 if start == divided_s[0] else 0
    return 0


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


def wmd(divided_s1, divided_s2):
    s1 = [w for w in divided_s1 if w not in stop_words]
    s2 = [w for w in divided_s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(divided_s1, divided_s2):
    s1 = [w for w in divided_s1 if w not in stop_words]
    s2 = [w for w in divided_s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    # words = str(s).lower().decode('utf-8')
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


def char_ngrams(n, word):
    return [word[i:i + n] for i in range(len(word)-n+1)]


def build_features(data, stops):
    X = pd.DataFrame()

    qs = pd.Series(data['question1'].tolist() + data['question2'].tolist())
    weights = calculate_tfidf(qs)
    del qs

    print('Building features')
    X['len_q1'] = data.question1.apply(word_len)   # 1:Length of Q1 str
    X['len_q2'] = data.question2.apply(word_len)   # 2:Length of Q2 str
    X['len_diff'] = abs(X.len_q1 - X.len_q2)   # 3:Length difference between Q1 and Q2

    print('Building char features')
    X['len_char_q1'] = data.q1_split.apply(word_len_char)  # 4:Char length of Q1
    X['len_char_q2'] = data.q2_split.apply(word_len_char)  # 5:Char length of Q2
    X['len_char_diff'] = data.apply(len_char_diff, axis=1, raw=True)  # 6:Char length difference between Q1 and Q2
    X['char_diff_unq_stop'] = data.apply(char_diff_unique_stop, stops=stops, axis=1, raw=True)  # 7: set(6)
    X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)  # 8:Char length Q1 / char length Q2

    print('Building word count features')
    X['word_count_q1'] = data.q1_split.apply(word_count)  # 9:Word count of Q1
    X['word_count_q2'] = data.q2_split.apply(word_count)  # 10:Word count of Q2
    X['word_count_diff'] = data.apply(wc_diff, axis=1, raw=True)  # 11:Word count difference between  Q1 and Q2
    X['word_count_ratio'] = data.apply(wc_ratio, axis=1, raw=True)  # 12:Word count Q1 / word count Q2

    X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)  # 13:Word count set(Q1 + Q2)
    X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)  # 14:Word count set(Q1) - word count set(Q2)
    X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)  # 15:Word count set(Q1) / word count set(Q2)

    X['total_unq_words_stop'] = data.apply(total_unq_words_stop, stops=stops, axis=1, raw=True)  # 16: 13 - stop words
    X['wc_diff_unique_stop'] = data.apply(wc_diff_unique_stop, stops=stops, axis=1, raw=True)  # 17: 14 - stop words
    X['wc_ratio_unique_stop'] = data.apply(wc_ratio_unique_stop, stops=stops, axis=1, raw=True)  # 18: 15 - stop words

    print('Building mark features')
    X['same_start'] = data.apply(same_start_word, axis=1, raw=True)  # 19 same start = 1 else = 0
    X['same_end'] = data.apply(same_end_word, axis=1, raw=True)  # 20 same end = 1 else = 0

    X['num_capital_q1'] = data.question1.apply(num_capital)  # 21
    X['num_capital_q2'] = data.question2.apply(num_capital)  # 22
    X['num_capital_diff'] = abs(X.num_capital_q1 - X.num_capital_q2)  # 23

    X['num_ques_mark_q1'] = data.question1.apply(num_ques_mark)  # 24
    X['num_ques_mark_q2'] = data.question2.apply(num_ques_mark)  # 25
    X['num_ques_mark_diff'] = abs(X.num_ques_mark_q1 - X.num_ques_mark_q2)  # 26

    print('Building another features')
    # 27 ~ 27+28(14*2)-1=54: First word in sentence(one hot)
    for start in common_start:
        X['start_%s_%s' % (start, 'q1')] = data.q1_split.apply(start_with, args=(start,))
    for start in common_start:  # 為了讓csv看起來更漂亮(更像one hot)
        X['start_%s_%s' % (start, 'q2')] = data.q2_split.apply(start_with, args=(start,))

    X['common_words'] = data.apply(common_words, axis=1, raw=True)  # 55:兩句相同的字數
    X['common_words_unique'] = data.apply(common_words_unit, axis=1, raw=True)  # 56:兩句相同的字母數

    X['word_match'] = data.apply(word_match_share, axis=1, raw=True)  # 57:字的重複比例 between Q1 and Q2
    X['word_match_stops'] = data.apply(word_match_share_stops, stops=stops,
                                       axis=1, raw=True)  # 58:字的重複比例 without stop word between Q1 and Q2
    X['tfidf_wm'] = data.apply(tfidf_word_match_share, weights=weights,
                               axis=1, raw=True)  # 59:字的重複比例 between Q1 and Q2 (TF-IDF值)
    X['tfidf_wm_stops'] = data.apply(tfidf_word_match_share_stops, stops=stops, weights=weights,
                                     axis=1, raw=True)  # 60:字的重複比例 without stop word between Q1 and Q2 (TF-IDF值)

    print('Building fuzzy features')
    # 61~67:Build fuzzy features
    X['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
                                         axis=1)
    X['fuzz_partial_token_set_ratio'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_partial_token_sort_ratio'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
    X['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                           axis=1)
    X['fuzz_token_sort_ratio'] = data.apply(
        lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    X['jaccard'] = data.apply(jaccard, axis=1, raw=True)  # 68:jaccard distance

    print('Build word2vec/glove distance features')
    # Build word2vec/glove distance features
    X['wmd'] = data.apply(lambda x: wmd(x['q1_split'], x['q2_split']), axis=1)  # 69
    X['norm_wmd'] = data.apply(lambda x: norm_wmd(x['q1_split'], x['q2_split']), axis=1)  # 70

    question1_vectors = np.zeros((data.shape[0], 300))

    print('Sent2Vec')
    # Sent2Vec
    for i, q in tqdm(enumerate(data.question1.values)):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors = np.zeros((data.shape[0], 300))
    for i, q in tqdm(enumerate(data.question2.values)):
        question2_vectors[i, :] = sent2vec(q)

    print('Building distance features')
    # Build distance features
    X['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                           np.nan_to_num(question2_vectors))]
    X['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                 np.nan_to_num(question2_vectors))]
    X['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                               np.nan_to_num(question2_vectors))]
    X['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                    np.nan_to_num(question2_vectors))]
    X['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

    X['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    X['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    X['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    X['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

    return X


if __name__ == '__main__':
    data_path = '/data1/quora_pair/50q_pair.csv'
    glove_path = '/data1/resources/GoogleNews-vectors-negative300.bin'
    out_path = '/data1/resources/quora_features.csv'

    df = pd.read_csv(data_path)
    df = df.fillna(' ')

    # 斷詞(中文的話這段要另外做斷詞)
    df['q1_split'] = df['question1'].map(lambda x: str(x).lower().split())
    df['q2_split'] = df['question2'].map(lambda x: str(x).lower().split())

    print('stop words:', stop_words)

    # Load glove model
    model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)
    norm_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=True)
    norm_model.init_sims(replace=True)

    # Build features
    df_new_feature = build_features(df, stop_words)

    # Save feature data to csv
    df_new_feature.to_csv(out_path, index=False)
