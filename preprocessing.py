import nltk
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
nltk.data.path.append('/home/ian/nltk_data')


def tokenizer(sentence):
    k = len(sentence)
    sentence = sentence.replace("-", "_")
    list_s = sentence.split(" ")
    for s_idx in range(len(list_s)):
        s = list_s[s_idx]
        words = s.split("_")
        if len(words) > 1:
            flag_not_word = False
            for w in words:
                k = len(w)
                t = wordnet.synsets(w)
                if len(w) == 1 or not wordnet.synsets(w):
                    flag_not_word = True
                    break
            if not flag_not_word:
                list_s.remove(s)
                s = s.replace("_", " ")
                list_s.insert(s_idx, s)
    return " ".join(list_s)


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for idx in range(len(chunked)):
        i = chunked[idx]
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    named_entity = " ".join(current_chunk)
    if named_entity not in continuous_chunk:
       continuous_chunk.append(named_entity)
    return continuous_chunk

SS = "I am a 19-year-old man who love state-of-the-art paper and F-14 flight"
TXT = "Barack Obama is the husband of Michelle Obama"
print(tokenizer(SS))
print(get_continuous_chunks(TXT))
