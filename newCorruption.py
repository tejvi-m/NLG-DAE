import spacy
import numpy as np
import pandas as pd
import utils
nlp = spacy.load("en_core_web_sm")

def test(x):

    #due to the mapping between words and integers the DAE does not deal
    #well with numbers it has not seen before.
    #in the training phase, the prices have been split into '£' and the value
    #while doing NER.
    #this is just a quick fix, but there has to be a real fix for the test phase at least
    try:
        return vocab_to_int[x]
    except:
        try:
            return vocab_to_int['£'+x]
        except:
            print(x)




def newCorrupt(sentence):
    #get names - propernounse and sometimes nouns.
    #get adjectives
    #NER will get the cardinal values i.e., numbers.
    #get everything in NER
    #finally make the whole thing a set.

    #using NER doesn't add much to it right now

    sentence = nlp(sentence)

    toString = set()


    for ent in sentence.ents:
        try:
            x = float(ent.text)
            num = True
        except:
            num = False

        if num == False:
            for word in ent.text.split(' '):
                toString.add((word))
        else:
            toString.add((ent.text))

    for token in sentence:
        pos = token.pos_
        add = True
        if pos == 'PROPN':
            toString.add((token.text))
        if pos == "ADJ" or pos == "ADV":
            toString.add((token.text))
        if pos == "NOUN":
            add = np.random.choice([1, 0, 0, 1])
            if add:
                toString.add((token.text))

    curroptedS = ''
    for string in toString:
        curroptedS += ' ' + string

    return curroptedS

if __name__ == "__main__":

    trainset = pd.read_csv('./data/trainset.csv')
    cor = trainset['ref'].apply(lambda x: newCorrupt(x))
    trainset = trainset.assign(corrupted=cor)
    as_tokens = trainset['corrupted'].apply(lambda x: [test(each) for each in x.split()])
    trainset = trainset.assign(tokenized_corrupted=as_tokens)


    trainset.to_csv('./data/processedTrainset.csv')





#
#
# sentence = ""
# import spacy
#
# nlp = spacy.load("en_core_web_sm")
# doc = nlp(sentence)
#
# for ent in doc.ents:
#     print(ent.text, ent.start_char, ent.end_char, ent.label_)
