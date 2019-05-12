#################################################################
# the following code is from https://github.com/mcleonard/NLG_Autoencoder/blob/master/train.py
# the code has some modifications from the original.
# code originally licensed by Mat Leonard under the MIT License
#################################################################

import os
import torch
import torch.nn as nn
import torch.optim as optim


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

trainset = pd.DataFrame()

trainset = pd.read_csv('drive/app/NLG/processedTrainset.csv', lineterminator='\n')

vocab_to_int, int_to_vocab = utils.get_tokens(trainset['clean'])

as_tokens = trainset['clean'].apply(lambda x: [vocab_to_int[each] for each in x.split()])
trainset = trainset.assign(tokenized=as_tokens)
trainset = trainset.assign(corrupted1=utils.replace_punctuation(trainset['corrupted']))
as_tokens = trainset['corrupted1'].apply(lambda x: [test(each) for each in x.split()])
trainset = trainset.assign(corrupted_tokenized=as_tokens)


def train(dataset, encoder, decoder, enc_opt, dec_opt, criterion,
          max_length=50, print_every=1000, plot_every=100,
          teacher_forcing=0.5, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = 0
    plot_losses = []
    losses = []
    for input_tensor, target_tensor in dataloader(dataset):
#         print("input_tensor: ", input_tensor)
        loss = 0
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        steps += 1

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        enc_opt.zero_grad()
        dec_opt.zero_grad()

        h, c = encoder.init_hidden(device=device)
        encoder_outputs = torch.zeros(max_length, 2*encoder.hidden_size).to(device)

        # Run input through encoder
        enc_outputs, enc_hidden = encoder.forward(input_tensor, (h, c))

        # Prepare encoder_outputs for attention
        encoder_outputs[:min(enc_outputs.shape[0], max_length)] = enc_outputs[:max_length,0,:]

        # First decoder input is the <SOS> token
        dec_input = torch.Tensor([[0]]).type(torch.LongTensor).to(device)
        dec_hidden = enc_hidden

        dec_outputs = []
        for ii in range(target_tensor.shape[0]):
          # Pass in previous output and hidden state
            dec_out, dec_hidden, dec_attn = decoder.forward(dec_input, dec_hidden, encoder_outputs)
            _, out_token = dec_out.topk(1)

            # Curriculum learning, sometimes use the decoder output as the next input,
            # sometimes use the correct token from the target sequence
            if np.random.rand() < teacher_forcing:
                dec_input = target_tensor[ii].view(*out_token.shape)
            else:
                dec_input = out_token.detach().to(device)  # detach from history as input

            dec_outputs.append(out_token)

            loss += criterion(dec_out, target_tensor[ii])

            # If the input is the <EOS> token (end of sentence)...
            if dec_input.item() == 1:
                break

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(encoder.parameters(), 5)
        nn.utils.clip_grad_norm_(decoder.parameters(), 5)

        enc_opt.step()
        dec_opt.step()

        print_loss_total += loss
        plot_loss_total += loss

        losses.append(loss)
        if steps % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Loss avg. = {print_loss_avg}")
            list1 = [int_to_vocab[each.item()] for each in input_tensor]
            list2 = [int_to_vocab[each.item()] for each in dec_outputs]
            list3 = [int_to_vocab[each.item()] for each in target_tensor]
            print(list1)
            print(list2)
            print("target: ", list3)

        print("steps: ", steps)
        if steps%5000 == 0:
            torch.save(encoder, "drive/app/NERnlgenc.pth")
            torch.save(decoder, "drive/app/NERnlgdec.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# max length for attention
max_length = 50

encoder = Encoder(len(vocab_to_int), hidden_size=512, drop_p=0.1).to(device)
decoder = Decoder(len(vocab_to_int), hidden_size=512, drop_p=0.1, max_length=max_length).to(device)

enc_opt = optim.Adam(encoder.parameters(), lr=0.001, amsgrad=True)
dec_opt = optim.Adam(decoder.parameters(), lr=0.001, amsgrad=True)
criterion = nn.NLLLoss()

if os.path.isfile('NERnlgenc.pth') and os.path.isfile('NERnlgdec.pth'):
    encoder = torch.load("nlgenc.pth", map_location = 'cpu')
    decoder = torch.load("nlgdec.pth", map_location = 'cpu')

epochs = 10

for e in range(1, epochs+1):
    print(f"Starting epoch {e}")
    train(trainset['tokenized'], encoder, decoder, enc_opt, dec_opt, criterion,
          teacher_forcing=0.9/e, device=device, print_every=200, save_every=1000,
          max_length=max_length)