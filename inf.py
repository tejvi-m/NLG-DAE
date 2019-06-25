import utils
from model import *
import pandas as pd
import torch
from torch import nn


file = open('concatenated.txt', 'r')
newFile = open('inferred_no.txt', 'w')
#the word chunks that serve as conditioning to the generations
L = file.readlines()
file.close()

count = 0

trainset = pd.read_csv('./data/trainset.csv')
trainset = trainset.assign(clean=utils.replace_punctuation(trainset['ref']))
vocab_to_int, int_to_vocab = utils.get_tokens(trainset['clean'])

encoder = torch.load("nlgenc.pth", map_location = 'cpu')
decoder = torch.load("nlgdec.pth", map_location = 'cpu')

def test(dataset, encoder, decoder,
          max_length=50, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for input_tensor in dataset:

        h, c = encoder.init_hidden(device=device)
        encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size).to(device)

        enc_outputs, enc_hidden = encoder.forward(input_tensor, (h, c))

        encoder_outputs[:min(enc_outputs.shape[0], max_length)] = enc_outputs[:max_length,0,:]

        dec_input = torch.Tensor([[0]]).type(torch.LongTensor).to(device)
        dec_hidden = enc_hidden

        dec_outputs = []

        for ii in range(max_length):
            dec_out, dec_hidden, dec_attn = decoder.forward(dec_input, dec_hidden, encoder_outputs)
            _, out_token = dec_out.topk(1)

            dec_input = out_token.detach().to(device)  # detach from history as input

            dec_outputs.append(out_token)

            if dec_input.item() == 1:
                break

        # list1 = [int_to_vocab[each.item()] for each in input_tensor]
        list2 = [int_to_vocab[each.item()] for each in dec_outputs]

        # print(list1)
        # print(list2)

        str2 = ''
        for i in list2:

            str2 = str2 + ' ' + i


        str2 = str2.replace('<COMMA>', ', ')
        str2 = str2.replace('<PERIOD>', '. ')
        str2 = str2.replace('<EOS>', '')
        str2 = str2.replace('<HYPHEN>', '-')
        str2 = str2.replace('<APOST>', '\'')

        #

        newFile.write(str2 + '\n')



df = pd.DataFrame({'col':L})

df = df.assign(clean=utils.replace_punctuation(df['col']))

as_tokens = df['clean'].apply(lambda x: [vocab_to_int[each] for each in x.split()])
df = df.assign(tokenized=as_tokens)
max_length = 50

for i in range(len(L)):
    inp = torch.Tensor(df['tokenized'][i]).view(-1, 1).type(torch.LongTensor)
    test([inp], encoder, decoder, device='cpu', max_length=max_length)
