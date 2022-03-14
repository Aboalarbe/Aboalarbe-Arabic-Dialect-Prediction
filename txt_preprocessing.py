import string
import re
from pyarabic import araby
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
import pickle
from keras.preprocessing.sequence import pad_sequences

with open('models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punctuation(text):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
    
def cleaning_pipeline(text):
    new_txt = re.sub('@[^\s]+','',str(text))
    new_txt = remove_emoji(new_txt)
    new_txt = re.sub(r'http\S+', '', new_txt)
    new_txt = remove_punctuation(new_txt)
    new_txt = re.sub(' +', ' ',new_txt)
    new_txt = re.sub(r'\s*[A-Za-z0-9]+\b', '' , new_txt)
    new_txt = new_txt.replace("\n" ," ")
    new_txt = re.sub(r'(.)\1+', r'\1', new_txt)
    tokens = araby.tokenize(new_txt)
    cleaned_tokens = [word for word in tokens if not word in stopwords.words('arabic')]
    return cleaned_tokens

def prepare_txt(text):
    sentence = [TreebankWordDetokenizer().detokenize(text)]
    txt_seq = tokenizer.texts_to_sequences(sentence)
    txt_padded = pad_sequences(txt_seq, maxlen=250)
    return txt_padded
	