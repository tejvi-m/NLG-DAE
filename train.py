#################################################################
# the following code is from https://github.com/mcleonard/NLG_Autoencoder/blob/master/train.py
# the code has some minor modifications.
# code originally licensed by Mat Leonard under the MIT License
#################################################################

import os
import torch
import torch.nn as nn
import torch.optim as optim


def dataloader(dataset, p_drop=0.6, max_length=50):

    # Corrupt dataset by randomly dropping words
    corrupted = utils.corrupt(dataset)
    # Shuffle words in each sequence
    shuffled = [utils.shuffle(seq, cor_seq) for seq, cor_seq in zip(dataset, corrupted)]

    for shuffled_seq, original_seq in zip(shuffled, dataset):
        # need to make sure our input_tensors have at least one element
        if len(shuffled_seq) == 0:
            shuffled_seq = [original_seq[np.random.randint(0, len(original_seq))]]

        input_tensor = torch.Tensor(shuffled_seq).view(-1, 1).type(torch.LongTensor)

        # Append <EOS> token to the end of original sequence
        target = original_seq.copy()
        target.append(1)
        target_tensor = torch.Tensor(target).view(-1, 1).type(torch.LongTensor)

        yield input_tensor, target_tensor

def train(dataset, encoder, decoder, enc_opt, dec_opt, criterion,
          max_length=50, print_every=1000, plot_every=100, save_every=5000,
          teacher_forcing=0.5, device=None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    steps = 0
    for input_tensor, target_tensor in dataloader(dataset):
        loss = 0
        print_loss_total = 0  # Reset every print_every

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
        encoder_outputs[:enc_outputs.shape[0]] = enc_outputs.squeeze()

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

        if steps % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(f"Loss avg. = {print_loss_avg}")
            print([int_to_vocab[each.item()] for each in input_tensor])
            print([int_to_vocab[each.item()] for each in dec_outputs])

        if steps % save_every == 0:
            torch.save(encoder, "./nlgenc.pth")
            torch.save(decoder, "./nlgdec.pth")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


trainset = pd.read_csv('.data/trainset.csv')
trainset = trainset.assign(clean=utils.replace_punctuation(trainset['ref']))
vocab_to_int, int_to_vocab = utils.get_tokens(trainset['clean'])
as_tokens = trainset['clean'].apply(lambda x: [vocab_to_int[each] for each in x.split()])
trainset = trainset.assign(tokenized=as_tokens)

# max length for attention
max_length = 50

encoder = Encoder(len(vocab_to_int), hidden_size=512, drop_p=0.1).to(device)
decoder = Decoder(len(vocab_to_int), hidden_size=512, drop_p=0.1, max_length=max_length).to(device)

enc_opt = optim.Adam(encoder.parameters(), lr=0.001, amsgrad=True)
dec_opt = optim.Adam(decoder.parameters(), lr=0.001, amsgrad=True)
criterion = nn.NLLLoss()

if os.path.isfile('nlgenc.pth') and os.path.isfile('nlgdec.pth'):
    encoder = torch.load("nlgenc.pth", map_location = 'cpu')
    decoder = torch.load("nlgdec.pth", map_location = 'cpu')

epochs = 10

for e in range(1, epochs+1):
    print(f"Starting epoch {e}")
    train(trainset['tokenized'], encoder, decoder, enc_opt, dec_opt, criterion,
          teacher_forcing=0.9/e, device=device, print_every=200, save_every=1000,
          max_length=max_length)
