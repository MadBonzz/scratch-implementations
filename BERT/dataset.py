import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class HindiData(Dataset):
    def __init__(self, sentences, tokenizer, mlm_probaility):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.mlm_probability = mlm_probaility
    def __len__(self):
        return len(self.sentences)
    
    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        prob_matrix = torch.rand_like(input_ids.float()) < self.mlm_probability
        special_tokens_mask = ~torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
            dtype=torch.bool
        )
        prob_matrix = prob_matrix & special_tokens_mask
        keep_same = torch.rand_like(input_ids.float()) < 0.1
        prob_matrix = prob_matrix & ~keep_same
        random_token = torch.rand_like(input_ids.float()) < 0.1
        random_token = random_token & prob_matrix
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, device=input_ids.device)
        input_ids = torch.where(random_token, random_tokens, input_ids)
        mask = prob_matrix & ~random_token
        input_ids[mask] = self.mask_token
        return input_ids, labels
    
    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = torch.tensor([self.sep_token_id] + self.tokenizer.encode(sent) + [self.bos_token_id], dtype=torch.long)
        input_ids, labels = self.mask_tokens(tokens)
        return {'input_ids' : input_ids, 'labels' : labels}

class DataCollatorWithDynamicPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding_value = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.padding_value)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.padding_value)
        
        return {'input_ids': padded_input_ids, 'labels': padded_labels}
