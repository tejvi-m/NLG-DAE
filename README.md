
# Unsupervised Natural Language Generation using Denoising Autoencoders

Built upon the PyTorch implementation of [Unsupervised Natural Language Generation with Denoising Autoencoders](https://arxiv.org/pdf/1804.07899.pdf) by Mat leonard - [https://github.com/mcleonard/NLG_Autoencoder](https://github.com/mcleonard/NLG_Autoencoder)

The original model outlined in the paper drops random (apart from not less frequent words) words to corrupt the input.

As a workaround and imrovement to many of the problems we observed with the original corruption method, this other version was built and it uses Spacy's NER and POS tagging to more closely simulate tabular data, while still being unsupervised.
