import utils
from model import *
import pandas as pd
import torch
import os
from torch import nn
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField

##############################################################################################
##################ML PIPELINE#################################################################

trainset = pd.read_csv('./data/trainset.csv')
trainset = trainset.assign(clean=utils.replace_punctuation(trainset['ref']))
vocab_to_int, int_to_vocab = utils.get_tokens(trainset['clean'])

encoder = torch.load("NERnlgenc.pth", map_location = 'cpu')
decoder = torch.load("NERnlgdec.pth", map_location = 'cpu')

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

        list1 = [int_to_vocab[each.item()] for each in input_tensor]
        list2 = [int_to_vocab[each.item()] for each in dec_outputs]

        print(list1)
        print(list2)

        str2 = ''
        for i in list2:

            str2 = str2 + ' ' + i


        str2 = str2.replace('<COMMA>', ', ')
        str2 = str2.replace('<PERIOD>', '. ')
        str2 = str2.replace('<EOS>', '')
        return str2


def generate(L):
    df = pd.DataFrame({'col':L})

    df = df.assign(clean=utils.replace_punctuation(df['col']))

    as_tokens = df['clean'].apply(lambda x: [vocab_to_int[each] for each in x.split()])
    df = df.assign(tokenized=as_tokens)

    inp = torch.Tensor(df['tokenized'][0]).view(-1, 1).type(torch.LongTensor)
    max_length = 50

    generation = test([inp], encoder, decoder, device='cpu',
              max_length=max_length)
    return generation

##########################################################################################



app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisnotasecret'


@app.route('/bleh', methods = ['GET', 'POST'])
def inputDesc():
    if form.validate_on_submit():
        text = request.form['text']
        processed_text = generate(text)
        return render_template('enterDetails.html', label = processed_text)

    return render_template('enterDetails.html', label = None)
app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('enterDetails.html', label = None)

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    print(text)
    try:
        processed_text = generate([text])
    except:
        processed_text = "Out of vocabulary word used. Try again."

    return render_template('enterDetails.html', label = processed_text, label1 = text)

from flask import Flask, make_response, request

def transform(text_file_contents):
    data = ""

    df = text_file_contents.split("\n")
    for cnt in range(1, len(df)-1):
        string = df[cnt]
        string.replace('yes', 'family friendly')
        string.replace('no', 'not family friendly')
        st = (string.split(","))[1:]
        string1 = ''
        for s in st:
            string1 += ' ' + s


        data+= generate([string1.strip()]) + '\n'




    return data


@app.route('/upload')
def form():
    return render_template("upload.html")
@app.route('/transform', methods=["POST"])
def transform_view():
    request_file = request.files['data_file']
    if not request_file:
        return "No file"
    file_contents = request_file.stream.read().decode("utf-8")

    result = transform(file_contents)

    response = make_response(result)
    response.headers["Content-Disposition"] = "attachment; filename=result.txt"
    return response

if __name__ == '__main__':
    app.run(debug = True)
