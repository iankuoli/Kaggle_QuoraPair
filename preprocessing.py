import re
import nltk
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
nltk.data.path.append('/data1/nltk_data')


def word_patterns_replace(text):
    # Transfer special chars
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    # text = re.sub('\&', " and ", text)  # 'and' is no in w2v but '&' have vec in w2v...
    text = re.sub("…", " ", text)
    text = re.sub("é", "e", text)

    # Remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)

    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
    text = re.sub('[0-9]+\.[0-9]+', " 1 ", text)

    # Clean shorthands
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r"e - mail", "e_mail", text)
    text = re.sub(r" j k ", " JK ", text)
    text = re.sub(r" J K ", " JK ", text)
    text = re.sub(r" J\.K\. ", " JK ", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.replace('?', ' ? ')
    text = text.replace(':', ' : ')
    text = text.replace(', ', ' , ')
    text = text.replace('. ', ' . ')
    text = text.replace('.\"', ' .\"')
    text = text.replace(',\"\",', ',\" \",')
    text = text.replace('(', ' ( ')
    text = text.replace(')', ' ) ')
    text = text.replace('\'s ', ' \'s ')
    text = text.replace('s\' ', ' s\' ')
    text = text.replace('n\'t ', ' not ')
    text = text.replace('\'m ', ' \'m ')

    # Detection symbol or tag
    text = text.replace('/', ' / ')
    text = re.sub(r"([\W]) / ([A-Za-z])", r"\1/\2", text)
    text = text.replace('<$', ' < $')
    text = re.sub(r"<([0-9])", r"< \1", text)

    #text = re.sub(r"[\W] / [\w]", "/", text)

    # Detect brief expression
    text = re.sub(r'((.[A-Z])+) \.', r'\1.', text)

    # Detect unit
    text = re.sub(r'([0-9])[M,m][H,h][Z,z]', r'\1 mhz ', text)
    text = re.sub(r'([0-9])[H,h][Z,z]', r'\1 hz ', text)
    text = re.sub(r'([0-9])[B,b][P,p][M,m]', r'\1 bpm ', text)
    text = re.sub(r'([0-9])[K,k][M,m] ', r'\1 km ', text)
    text = re.sub(r'([0-9])[C,c][M,m] ', r'\1 cm ', text)
    text = re.sub(r'([0-9])[K,k][G,g] ', r'\1 kg ', text)
    text = re.sub(r'([0-9])[K,k][G,g][S,s] ', r'\1 kgs ', text)
    text = re.sub(r'([0-9])[M,m][G,g] ', r'\1 mg ', text)
    text = re.sub(r'([0-9])[M,m][L,l] ', r'\1 ml ', text)
    text = re.sub(r'([0-9])[M,m][S,s] ', r'\1 ms ', text)
    text = re.sub(r'([0-9])[L,l][P,p][A,a] ', r'\1 kg ', text)
    text = re.sub(r'\$([0-9])', r'$ \1 ', text)
    text = re.sub(r'([0-9]) [V,v]', r'\1 volt ', text)
    text = re.sub(r'([0-9])[V,v]', r'\1 volt ', text)
    text = re.sub(r'([0-9])\-[V,v]', r'\1 volt ', text)
    text = re.sub(r'([0-9])[K,k][V,v][A,a]', r'\1 kVA ', text)
    text = re.sub(r'([0-9])[K,k][P,p][H,h]', r'\1 kph ', text)
    text = re.sub(r'([0-9])[M,m][P,p][H,h]', r'\1 mph ', text)
    text = re.sub(r'([0-9])hours', r'\1 hours ', text)
    text = re.sub(r'([0-9])hour', r'\1 hour ', text)

    # Digit expression
    text = re.sub(r'([0-9]),([0-9])', r'\1\2', text)
    text = re.sub(r'([0-9])[K,k] ', r'\g<1>000 ', text)
    text = re.sub(r'([0-9])\+([0-9])', r'\1 + \2', text)

    text = text.replace('  ', ' ')

    text = tokenizer(text)

    text = re.sub(r"\-", " - ", text)

    return text


def tokenizer(sentence):
    k = len(sentence)
    sentence = sentence.replace("-", "_")
    sentence = sentence.replace('–', '_')
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
# SS ="I'm a 19-year-old. I spend $5000 which is 20% of my 9/11 sayings… . Aréaééé 0.999 5kgs 10kg 555kph 5hours e-mail e - mail? jj korea J.K."
# SSS = "'online' ‘god’ 'suroor' 'non aspiration' dyke” 'dm ''biba'' 'couldn't've 'mia mia' if “膜”  ‘weekends’ player’s ‘would’ ‘would ''how “her”"
#SS = "I am a 19-year-old. man who love neural-based paper and F-14 flight"
#TXT = "Barack Obama is the husband of Michelle Obama"
# print(tokenizer(SS))
# print(word_patterns_replace(SS)+'\n')
# print(word_patterns_replace(SSS)+'\n')
#print(get_continuous_chunks(TXT))
#TXT = "\"Ted's Indian-made <20K 10 V dicks that cost <$10,000 hasn't been detected/protected at (9+2)/11 with [/math] and MOOCs/E-learning (900/1,800 bpm Tu-95).\n PIF: 14-years-old Trump–Clinton U.S. Presidential debate is good for 10Km? <\html>\""
#TXT = "What is the output for in main {char *ptr=""hello""; ptr [0] ='m'; printf (""%s"" , *s);} ?"
#TXT = "If light has zero mass , then as per this [math]E=mc^2[/math] , light must have zero energy. Is it so ?"
#TXT = "What is the story of Kohinoor (Koh-i-Noor) Diamond"
# TXT="How would I find the necessary number of turns on a transformer primary if the secondary voltage required is 120 V at 60 Hz ?"
TXT="A distribution transformer is rated at 18 kVA , 20,000/480 V , and 60 hz. can this transformer safely supply 15kVA to a 415-V load at 50hz? Why or not?"
print('\nINPUT:\n' + TXT)
print('\nOUTPUT:\n' + word_patterns_replace(TXT))

