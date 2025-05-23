from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import numpy as np

data_path = "C:\\Users\\shour\\OneDrive - vit.ac.in\\torch-implementations\\tokenizer.txt"

with open(data_path, 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences]

corpus = "".join(sentences)

chars = set(corpus)

unk_token = '[UNK]'
mask_token = '[MASK]'
pad_token = '[PAD]'
bos_token = '[BOS]'
sep_token = '[SEP]'

special_dict = {'unk_token' : unk_token, 'mask_token' : mask_token, 'pad_token' : pad_token, 'bos_token' : bos_token, 'sep_token' : sep_token}

tokenizer = Tokenizer(BPE(byte_fallback=True))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=1019, min_frequency=5, show_progress=True,end_of_word_suffix="_", initial_alphabet=list(chars))

def batch_iterator(batch_size=500):
    for i in range(0, len(corpus), batch_size):
        yield corpus[i:i+batch_size]

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
fast_tokenizer.add_special_tokens(special_dict)

fast_tokenizer.save_pretrained('bert-tiny-hindi')