import re


def get_contractions(text):
    contractions_dictionary = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have", "'cause": "because",
        "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
        "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
        "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
        "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
        "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have",
        "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
        "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
        "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
        "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
        "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have",
        "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
        "they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
        "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
        "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
        "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
        "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have",
        "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
        "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
        "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
        "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
        "y'all": "you all", "y'all'd": "you all would", "y'alll'd've": "you all would have", "y'all're": "you all are",
        "y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
        "you'll've": "you will have", "you're": "you are", "you've": "you have", "isnt": "is not", " im ": " i am ",
        "amp": '', "&amp": '', '--&gt': '', '&gt': '', '♥': '', '🤷': '',
        "aren’t": "are not", "can’t": "cannot", "can’t’ve": "cannot have", "’cause": "because",
        "could’ve": "could have", "couldn’t": "could not", "couldn’t’ve": "could not have", "didn’t": "did not",
        "doesn’t": "does not", "don’t": "do not", "hadn’t": "had not", "hadn’t’ve": "had not have",
        "hasn’t": "has not", "haven’t": "have not", "he’d": "he would", "he’d’ve": "he would have", "he’ll": "he will",
        "he’ll’ve": "he will have", "he’s": "he is", "how’d": "how did", "how’d’y": "how do you", "how’ll": "how will",
        "how’s": "how is", "i’d": "i would", "i’d’ve": "i would have", "i’ll": "i will", "i’ll’ve": "i will have",
        "i’m": "i am", "i’ve": "i have", "isn’t": "is not", "it’d": "it would", "it’d’ve": "it would have",
        "it’ll": "it will", "it’ll’ve": "it will have", "it’s": "it is", "let’s": "let us", "ma’am": "madam",
        "mayn’t": "may not", "might’ve": "might have", "mightn’t": "might not", "mightn’t’ve": "might not have",
        "must’ve": "must have", "mustn’t": "must not", "mustn’t’ve": "must not have", "needn’t": "need not",
        "needn’t’ve": "need not have", "o’clock": "of the clock", "oughtn’t": "ought not",
        "oughtn’t’ve": "ought not have", "shan’t": "shall not", "sha’n’t": "shall not", "shan’t’ve": "shall not have",
        "she’d": "she would", "she’d’ve": "she would have", "she’ll": "she will", "she’ll’ve": "she will have",
        "she’s": "she is", "should’ve": "should have", "shouldn’t": "should not", "shouldn’t’ve": "should not have",
        "so’ve": "so have", "so’s": "so is", "that’d": "that would", "that’d’ve": "that would have",
        "that’s": "that is", "there’d": "there would", "there’d’ve": "there would have", "there’s": "there is",
        "they’d": "they would", "they’d’ve": "they would have", "they’ll": "they will", "they’ll’ve": "they will have",
        "they’re": "they are", "they’ve": "they have", "to’ve": "to have", "wasn’t": "was not",
        "we’d": "we would", "we’d’ve": "we would have", "we’ll": "we will", "we’ll’ve": "we will have",
        "we’re": "we are", "we’ve": "we have", "weren’t": "were not", "what’ll": "what will",
        "what’ll’ve": "what will have", "what’re": "what are", "what’s": "what is", "what’ve": "what have",
        "when’s": "when is", "when’ve": "when have", "where’d": "where did", "where’s": "where is",
        "where’ve": "where have", "who’ll": "who will", "who’ll’ve": "who will have", "who’s": "who is",
        "who’ve": "who have", "why’s": "why is", "why’ve": "why have", "will’ve": "will have", "won’t": "will not",
        "won’t’ve": "will not have", "would’ve": "would have", "wouldn’t": "would not", "wouldn’t’ve": "would not have",
        "y’all": "you all", "y’all’d": "you all would", "y’alll’d’ve": "you all would have", "y’all’re": "you all are",
        "y’all’ve": "you all have", "you’d": "you would", "you’d’ve": "you would have", "you’ll": "you will",
        "you’ll’ve": "you will have", "you’re": "you are", "you’ve": "you have"
    }

    contraction_re = re.compile('(%s)' % '|'.join(contractions_dictionary.keys()))

    def replace(match):
        return contractions_dictionary[match.group(0)]

    return contraction_re.sub(replace, text)


# Stop words to not delete
ok_stop_words = {'not'
                 }

# Unwanted tokens to be removed
unwated_tokens = {'st', 'nd', 'rd', 'lambda', 'alpha',
                  }

# Positive Emoticons
emoticons_pos_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D',
                       '8-D', '8D', '=-3', '=3', ':-))', ":'-)", ":')"}
emoticons_pos_laugh = {'x-D', 'xD', 'X-D', 'XD', '=-D', '=D'}
emoticons_pos_kiss = {':*', ':^*'}
emoticons_pos_playfull = {'>:P', ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)',
                          '>:-)'}
emoticons_pos_love = {'<3'}

# Negative Emoticons
emoticons_neg_sad = {':L', ':-/', '>:/', ':S', '>:[', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                     '>:(', ':(', '>.<'}
emoticons_neg_angry = {':@'}
emoticons_neg_cry = {":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}


def emoticon_translation(word):
    newword = ''
    if word in emoticons_pos_happy:
        newword = 'happy'
    elif word in emoticons_pos_laugh:
        newword = 'laugh'
    elif word in emoticons_pos_kiss:
        newword = 'kiss'
    elif word in emoticons_pos_playfull:
        newword = 'playfull'
    elif word in emoticons_pos_love:
        newword = 'love'
    elif word in emoticons_neg_sad:
        newword = 'sad'
    elif word in emoticons_neg_angry:
        newword = 'angry'
    elif word in emoticons_neg_cry:
        newword = 'cry'
    # if newword != '':
    #    print(':)')
    return newword