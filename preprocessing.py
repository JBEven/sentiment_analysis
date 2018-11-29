#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:09:50 2018

@author: leli
"""

import re

def replaceThreeOrMore(text):
    
    """
    look for 3 or more repetitions of character (and newline) and replace with the character itself
    eg: 'hahahah' --- 'hahah'
        'rougeee' --- 'rougee'
        '😊😊😊'   --- '😊😊'
    
    """
    
    pattern = re.compile(r"(.+)\1{2,}", re.DOTALL)
    return pattern.sub(r"\1\1", text)


#text = "agcddd u'\U0001f6bb'u'\U0001f6bb'u'\U0001f6bb' hahaha www:'http: " 
#print (replaceThreeOrMore(text))


def replace_non_alphanumeric_begin_sentence(text):
    """
    replace any text that begins with non [a-zA-Z0-9] character with its
    original
    eg: '###abc asd' ---> 'abc asd'
        '.abc efg ---> abc efg'
    """
    return re.sub(r"^[^A-Za-z0-9]+(.*)", r"\1", text)


#print (replace_non_alphanumeric_begin_sentence("...abc paris "))

def replace_url_by(text, entity = 'URL'):
    
    """
    Convert www.* or https?://* to URL and return a 2-elements tuple.
    
    Eg: 
    >> text = "les liens sont: http://abc.com et www.hotmail.com"
    >> replace_url_by(text)
    >> ('les liens sont: URL et URL', [('http://abc.com', 'URL'), ('www.hotmail.com', 'URL')])

    
    
    """
    
    all_url = re.findall(r'((www[]*\.[]*[^\s]+)|(http[s ]*(?:[/\.]*)[^\s]*))', text)
    url_tuple = [(url[0], entity) for url in all_url]
    
    
    return (re.sub(r'((www[]*\.[]*[^\s]+)|(http[s ]*(?:[/\.]*)[^\s]*))', entity, text), url_tuple)



def replace_email_by(text, entity = 'EMAIL'):
    """
    Convert email abc@disney.com to EMAIL and return a 2-elements tuple.
    
    Eg: 
    >> text = "Ton email est: abc@gmail.com et son email est: efg@hotmail.com"
    >> replace_email_by(text)
    >> ('Ton email est: EMAIL et son email est: EMAIL',[('abc@gmail.com', 'EMAIL'), ('efg@hotmail.com', 'EMAIL')])
    
    """
    
    all_emails = re.findall(r"([a-zA-Z0-9_.+-]+(@|\[at\]|\(at\))[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    email_tuple = [(eml[0], entity) for eml in all_emails] 
    
    return (re.sub(r"([a-zA-Z0-9_.+-]+(@|\[at\]|\(at\))[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", entity, text), email_tuple)

#print (replace_email_by("Mon email addresse est:  abc@disneyland.fr")) 



def replace_emoji_by(text, entity = 'EMOJI'):
    
    """
    Convert emoji 😊 to EMOJI and return a 2-elements tuple.
    
    Eg: 
    >> text = "Merci 😊👍"
    >> replace_emoji_by(text)
    >> ('Merci EMOJIEMOJI', [('😊', 'EMOJI'), ('👍', 'EMOJI')])
    """
    
    
    
    emojis = [(char, entity) for char in casual_tokenize(text) if char in list(emoji.UNICODE_EMOJI.keys())]
    return (u''.join(entity if char in list(emoji.UNICODE_EMOJI.keys()) else char for char in text), emojis)

    
#print (replace_emoji_by("Merci 😊👍", entity = 'EMOJI'))


def remove_tag_mark(text):
    """
    remove tag mark:
    
    Eg:
    '@abc je viens de ...' >>> 'je viens de ...'
    '@ abc je viens de ...' >>> 'je viens de ...'
        
    """
    
    return (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()))
    







def add_space_to_punct(text):
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        text = text.replace(char, ' ' + char + ' ')
    return text
    
    
    
def replace_voyelle_with_accent(text):
        voyelle_dic = {'à' : 'a',
                       'â' : 'a',
                       'é' : 'e', 
                       'è' : 'e', 
                       'ê' : 'e', 
                       'ë' : 'e', 
                       'ù' : 'u', 
                       'ô' : 'o',
                       'û' : 'u'}
        
        for v in voyelle_dic.keys():
            text = text.replace(v, voyelle_dic[v])
        return text
    


    





def preprocess(text):
    
    try:
        text = remove_tag_mark(text)
        text = replace_non_alphanumeric_begin_sentence(text)
        text = replace_url_by(text)[0]   ##replace_url_by should be before replaceThreeOrMore otherwise www. ---> ww.\n",
        text = replace_email_by(text)[0]
        text = replaceThreeOrMore(text)  ## put this one at the end since www.abc.com\n",
        #text = replace_voyelle_with_accent(text)
        text = add_space_to_punct(text)
        text = text.lower()
    except:
        pass
    return text
    
    
    
    
    
def get_ith_turn_index(df, ith, agent = 0):
    
    if agent == 0:
        grouped_df = df[df['agent'] == 'visitor'].groupby(['conv_uid'])
    elif agent == 1:
        grouped_df = df[df['agent'] == 'operator'].groupby(['conv_uid'])
    elif agent == 2:
        grouped_df = df.groupby(['conv_uid'])
    else:
        print ('agent value not valid')
    
    turns = []
    
    for i in grouped_df.groups.keys():
        try:
            messages = grouped_df.get_group(i)
            if len(messages) >= ith:
                turns.append(messages.iloc[[ith-1]].index.values[0])
        except:
            pass
        
    return turns


from scipy import stats
def symmetric_KL_distance(p, q):
    p = p + 10**(-8)
    q = q + 10**(-8)
    avg = (p+q)/2
    return 0.5*(stats.entropy(p, avg) + stats.entropy(q, avg))
    
    

    

def transformText(text, stops):
    
    #stops = set(stopwords.words("english"))
    
    # Convert text to lower
    text = text.lower()
    # Removing non ASCII chars    
    #text = re.sub(r'[^\x00-\x7f]',r' ',text)
    
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    
    # Removing all the stopwords
    filtered_words = [word for word in text.split() if word not in stops]
    
    # Removing all the tokens with lesser than 2 characters
    filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=2)
    
    # Preprocessed text after stop words removal
    text = " ".join(filtered_words)
    
    # Remove the punctuation
    text = gensim.parsing.preprocessing.strip_punctuation2(text)
    
    # Strip all the numerics
    #text = gensim.parsing.preprocessing.strip_numeric(text)
    
    # Strip multiple whitespaces
    text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
    
    # Stemming
    #return gensim.parsing.preprocessing.stem_text(text)
    return text


