with open('corpus.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

sentences = [sentence.strip() for sentence in sentences]

corpus = ' '.join(sentences)
chars = sorted(set(corpus))

class WordPeice:
    def __init__(self, vocab_size : int, min_freq : float, initial_alphabet : list[chr] = None, byte_fallback : bool = False, unk_token : str = None):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.byte_fallback = byte_fallback
        self.unk_token = unk_token
        self.chr_to_idx = {}
        if self.byte_fallback == False and self.unk_token is None:
            raise ValueError("Need a UNK token in case byte fallback is set False. Either set byte_fallback=True or provide a unk_token")
        if initial_alphabet is not None:
            for idx, char in enumerate(initial_alphabet):
                self.chr_to_idx[char] = idx
        if not self.byte_fallback:
            curr_len = len(self.chr_to_idx.keys())
            self.chr_to_idx[unk_token] = curr_len

    def get_stats(self, tokens : list):
        pair_freqs = {}
        for pair in zip(tokens, tokens[1:]):
            pair_freqs[pair] = pair_freqs.get(pair, 0) + 1
        for key in pair_freqs.keys():
            pair_freqs[key] /= (tokens.count(key[0]) * tokens.count(key[1]))
        return pair_freqs

    def merge(self, tokens: list, pair: tuple[int, int], new_token_id: int):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_token_id)  
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def initial_encoding(self, corpus : str):
        s = list(corpus)
        tokens = []
        for char in s:
            if char in self.chr_to_idx.keys():
                tokens.append(self.chr_to_idx[char])
            else:
                char_tokens = char.encode('utf-8')
                curr_len = len(self.chr_to_idx.keys())
                for idx, hex_char in enumerate(char_tokens):
                    self.chr_to_idx[hex_char] = idx + curr_len
                    tokens.append(idx + curr_len)
        return tokens
            
    def get_vocab(self):
        return self.chr_to_idx
    
    def train(self, corpus : str):
        tokens = self.initial_encoding(corpus)
        print("Got initial tokens")
        curr_len = len(self.chr_to_idx.keys())
        print(f"Length of vocab dict is : {curr_len}")
        while curr_len < self.vocab_size:
            pair_freqs = self.get_stats(tokens)
            highest_pair = max(pair_freqs, key=pair_freqs.get)
            freq = pair_freqs[highest_pair]
            if freq < self.min_freq:
                break
            self.chr_to_idx[highest_pair] = curr_len
            tokens = self.merge(tokens, highest_pair, curr_len)
            print(f"Merging tokens : {highest_pair} to form new token : {curr_len}")
            curr_len += 1
        print(f"Length of vocab dict is : {len(self.chr_to_idx.keys())}")

    def encode(self, s : str):
        s = list(s)
        tokens = []
        for char in s:
            if char in self.chr_to_idx.keys():
                tokens.append(self.chr_to_idx[char])
            elif char.encode('utf-8') in self.chr_to_idx.keys():
                tokens.append(self.chr_to_idx[char.encode('utf-8')])
            else:
                if self.byte_fallback:
                    char_tokens = char.encode('utf-8')
                    curr_len = len(self.chr_to_idx.keys())
                    for idx, _ in enumerate(char_tokens):
                        tokens.append(idx + curr_len)
                else:
                    tokens.append(self.chr_to_idx[self.unk_token])
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.chr_to_idx.get(p, float("inf")))
            if pair not in self.chr_to_idx:
                break
            idx = self.chr_to_idx[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

bpe = WordPeice(2048, 0.02, initial_alphabet=chars, unk_token='UNK')
bpe.train(corpus[:1000])
tokens = bpe.encode(corpus[:100])
print(tokens)
print(len(tokens))