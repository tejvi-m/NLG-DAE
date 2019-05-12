def dataloader(dataset, p_drop=0.6, max_length=50):


    shuffled = [utils.shuffle(seq, cor_seq) for seq, cor_seq in zip(trainset['tokenized'], trainset['corrupted_tokenized'] )]

    for shuffled_seq, original_seq in zip(shuffled, trainset['tokenized']):
        # need to make sure our input_tensors have at least one element
        if len(shuffled_seq) == 0:
            shuffled_seq = [original_seq[np.random.randint(0, len(original_seq))]]

        try:
          input_tensor = torch.Tensor(shuffled_seq).view(-1, 1).type(torch.LongTensor)
        except:
          input_tensor = original_seq.copy()
          input_tensor = torch.Tensor(input_tensor).view(-1, 1).type(torch.LongTensor)

        # Append <EOS> token to the end of original sequence
        target = original_seq.copy()
        target.append(1)
        target_tensor = torch.Tensor(target).view(-1, 1).type(torch.LongTensor)

        yield input_tensor, target_tensor
