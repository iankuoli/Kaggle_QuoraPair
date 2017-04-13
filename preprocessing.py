import re
import nltk
from nltk.corpus import wordnet
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
nltk.data.path.append('/home/ian/nltk_data')


def word_patterns_replace(text):
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

    # detection symbol or tag
    text = text.replace('/', ' / ')
    text = re.sub(r"([\W]) / ([A-Za-z])", r"\1/\2", text)
    text = text.replace('<$', ' < $')
    text = re.sub(r"<([0-9])", r"< \1", text)

    #text = re.sub(r"[\W] / [\w]", "/", text)

    # detect brief expression
    text = re.sub(r'((.[A-Z])+) \.', r'\1.', text)

    # detect unit
    text = re.sub(r'([0-9])[M,m][H,h][Z,z]', r'\1 mhz', text)
    text = re.sub(r'([0-9])[H,h][Z,z]', r'\1 hz', text)
    text = re.sub(r'([0-9])[B,b][P,p][M,m]', r'\1 bpm', text)
    text = re.sub(r'([0-9])[K,k][M,m] ', r'\1 km ', text)
    text = re.sub(r'([0-9])[C,c][M,m] ', r'\1 cm ', text)
    text = re.sub(r'([0-9])[K,k][G,g] ', r'\1 kg ', text)
    text = re.sub(r'([0-9])[M,m][G,g] ', r'\1 mg ', text)
    text = re.sub(r'([0-9])[M,m][L,l] ', r'\1 kg ', text)
    text = re.sub(r'([0-9])[L,l][P,p][A,a] ', r'\1 kg ', text)
    text = re.sub(r'\$([0-9])', r'$ \1', text)
    text = re.sub(r'([0-9]) [V,v]', r'\1 volt', text)
    

    # digit expression
    text = re.sub(r'([0-9]),([0-9])', r'\1\2', text)
    text = re.sub(r'([0-9])[K,k] ', r'\g<1>000 ', text)
    text = re.sub(r'([0-9])\+([0-9])', r'\1 + \2', text)

    text = text.replace('  ', ' ')

    text = tokenizer(text)

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

#SS = "I am a 19-year-old man who love neural-based paper and F-14 flight"
#TXT = "Barack Obama is the husband of Michelle Obama"
#print(tokenizer(SS))
#print(get_continuous_chunks(TXT))
#TXT = "\"Ted's Indian-made <20K 10 V dicks that cost <$10,000 hasn't been detected/protected at (9+2)/11 with [/math] and MOOCs/E-learning (900/1,800 bpm Tu-95).\n PIF: 14-years-old Trump–Clinton U.S. Presidential debate is good for 10Km? <\html>\""
#TXT = "What is the output for in main {char *ptr=""hello""; ptr [0] ='m'; printf (""%s"" , *s);} ?"
#TXT = "If light has zero mass , then as per this [math]E=mc^2[/math] , light must have zero energy. Is it so ?"
#TXT = "What is the story of Kohinoor (Koh-i-Noor) Diamond"
TXT="How would I find the necessary number of turns on a transformer primary if the secondary voltage required is 120 V at 60 Hz ?"
TXT="A distribution transformer is rated at 18 kVA , 20,000/480 V , and 60 hz. can this transformer safely supply 15kVA to a 415-V load at 50hz ? Why or not ?"
print('\nINPUT:\n' + TXT)
print('\nOUTPUT:\n' + word_patterns_replace(TXT))

