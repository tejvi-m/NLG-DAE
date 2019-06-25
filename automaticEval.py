#check grammar
#check hallucination
#check conditioning
import spacy
import numpy as np
import pandas
import utils
from newCorruption import *

#also have a bunch of synonyms i guess. Use some kind of semantic
#vectorization for it?

nlp = spacy.load("en_core_web_sm")

texts = open('refs.txt', 'r').readlines()
inputs = open('concatenated.txt', 'r').readlines()
geners = open('inferred_no.txt', 'r').readlines()
halls = []
conds = []

def getVals(inp, ref, gen):

    #input is the corrupted one
    #text is reference text
    #gen is gen

    text = ref


    textlen = len(gen)
    Gen = gen
    text = nlp(text)
    gen = nlp(gen)

    #hallucination
    hallucination = 0
    print(text.ents, gen.ents)
    for ent in gen.ents:
        if ent not in text.ents:
            print(ent.text)
            hallucination += 1

    hallucination1 = 0
    textNouns = []
    genNouns = []
    textAdj = []
    genAdj = []

    print(inp)
    for token in gen:
        if token.pos_ == 'PROPN' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            genNouns.append(token)


    for token in text:

        if token.pos_ == 'PROPN' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            textNouns.append(token)


    print(genNouns, genAdj)
    print(textNouns, textAdj)

    hallucination1 = 0

    for word in genNouns:
        cost = 1
        if word not in textNouns:
            hallucination1 += cost

    print(hallucination1/textlen, hallucination)

    cond = len(inp)

    for i in inp.split(" "):
        if i not in Gen:
            print(i)
            cond -= 1

    print(cond/len(inp))
    halls.append(hallucination1/textlen)
    conds.append(cond/len(inp))

for i in range(600):
    getVals(inputs[i], texts[i], geners[i])

print(len(halls), len(conds))
print(sum(halls)/len(halls), sum(conds)/len(conds))
