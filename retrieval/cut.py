import jieba
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

is_char_ = True
print('is_char',is_char_)

def cut( x):
    # 分词和分字
    is_char = is_char_
    sentence = []
    if is_char:
        if isinstance(x, str):
            line = x.strip()
            if line != '':
                temp = []
                for each in jieba.lcut(line):
                    if each != ' ':
                        # 判断 中文 还是英文#
                        patterns_ch = re.compile(r'[^\u4e00-\u9fa5$]', flags=re.I)
                        is_ch = re.findall(patterns_ch, each)
                        if not is_ch:
                            for char in each:
                                temp.append(char.strip())
                        elif each != '\n':
                            temp.append(each)
                if temp != []:
                    sentence = temp
    else:
        if isinstance(x, str):
            line = x.strip()
            if line != '':
                sentence = ' '.join(jieba.lcut(x)).split()
    return sentence



def cut_en(sentence):
    #split english 小写，分词，词性还原


    # 获取单词的词性
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    tokens = word_tokenize(sentence.lower())  # 小写,分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
    return lemmas_sent

