from underthesea import word_tokenize, sent_tokenize, pos_tag
import regex

def removeStopWords(text, stop_words):
    text = text.split()
    result = []
    for word in text:
        if word not in stop_words:
            result.append(word)
    return ' '.join(result)

def wordtokenize(text):
    return word_tokenize(text, format="text")

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.', '')
        ## POS tag
        lst_word_type = ['N', 'NP', 'A', 'AB' , 'AY' , 'V', 'VB', 'VY', 'R' , 'M']
        # lst_word_type = [ 'N', 'NP', 'A', 'AB' , 'AY' , 'ABY', 'V', 'VB', 'VY', 'R' , 'M', 'I']
        # print(pos_tag(word_tokenize(sentence, format="text")))

        sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(word_tokenize(sentence, format="text")))
        new_document = new_document + sentence + ' '
    # Delete excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def removeSpecialChar(text):
    return regex.sub(r'[^\w\s]', '', text)

def removeSpecialCharAdvance(textStr):
    textSplited = textStr.split()
    textSplited = [[regex.sub('[0-9]+','', e) for e in text] for text in textSplited] # số
    textSplited = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml']] for text in  textSplited] # ký tự đặc biệt
    # print(textSplited)
    return ' '.join([''.join(text) for text in textSplited])

def stepByStep(text):
    with open('data/vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split('\n')

    text = str(text)
    # text = wordtokenize(text.lower())
    text = process_postag_thesea(text.lower())
    text = removeSpecialChar(text)
    text = removeStopWords(text, stop_words)
    return text

# text = "Áo Ba Lỗ"
# text = stepByStep(text)
# print(text)
# print(type(text))