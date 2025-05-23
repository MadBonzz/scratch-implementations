from tqdm import tqdm
import math

class UnigramModel:
    def __init__(self, k = 1):
        self.K = k
        self.counts_dict = {}
        self.start_dict = {}
        self.end_dict = {}
        self.V = 0
        self.N = 0
        self.N_SENTENCES = 0
        self.starts = 0
        self.ends = 0

    def train(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            sentences = file.readlines()
        words = set()
        for i in range(len(sentences)):
            sentence = sentences[i].strip()
            words = words.union(set(sentence.split()))
            sentences[i] = sentence.split()
        for sentence in tqdm(sentences):
            for i in range(len(sentence)):
                if i == 0:
                    self.start_dict[sentence[i]] = self.start_dict.get(sentence[i], 0) + 1
                elif i == len(sentence) - 1:
                    self.end_dict[sentence[i]] = self.end_dict.get(sentence[i], 0) + 1
                else:
                    self.counts_dict[sentence[i]] = self.counts_dict.get(sentence[i], 0) + 1
                    self.N += 1        
        self.V = len(words)
        self.N_SENTENCES = len(sentences)
        self.starts = len(self.start_dict.keys())
        self.ends = len(self.end_dict.keys())
        
    def calculate_scores(self, word):
        return (self.counts_dict.get(word, 0) + self.K) / (self.N + self.K * self.V)
    
    def calculate_start_scores(self, word):
        return (self.start_dict.get(word, 0) + self.K) / (self.N_SENTENCES + self.K * self.starts)

    def calculate_end_scores(self, word):
        return (self.end_dict.get(word, 0) + self.K) / (self.N_SENTENCES + self.K * self.ends)
    
    def get_log_likelihood(self, sentence):
        words = sentence.split()
        scores = []
        total_score = 0
        probs = 1
        score = self.calculate_start_scores(words[0])
        total_score += math.log(score)
        probs *= score
        scores.append(score)
        for i in range(1, len(words) - 1):
            score = self.calculate_scores(words[i])
            total_score += math.log(score)
            probs *= score
            scores.append(score)
        score = self.calculate_end_scores(words[-1])
        total_score += math.log(score)
        probs *= score
        scores.append(score)
        total_score /= len(words)
        perplexity = 1 / math.pow(probs, 1 / len(scores))
        return scores, total_score, perplexity

class NGramModel:
    def __init__(self, n, start_token, end_token, k=1):
        self.n = n
        self.K = k
        self.start_token = start_token
        self.end_token = end_token
        self.counts_dict = None
        self.precursor_dict = None
        self.V = 0

    def train(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            sentences = file.readlines()
        words = set()
        for i in range(len(sentences)):
            sentence = sentences[i].strip()
            for j in range(self.n-1):
                sentence = self.start_token + " " + sentence
            sentence += " " + self.end_token
            words = words.union(set(sentence.split()))
            sentences[i] = sentence.split()
        counts_dict = {}
        precursor_dict = {}
        for sentence in tqdm(sentences):
            for i in range(len(sentence) - self.n):
                n_gram = sentence[i:i+self.n]
                key = " ".join(n_gram[:-1])
                n_gram = " ".join(n_gram)
                counts_dict[n_gram] = counts_dict.get(n_gram, 0) + 1
                precursor_dict[key] = precursor_dict.get(key, 0) + 1
        self.counts_dict = counts_dict
        self.precursor_dict = precursor_dict
        self.V = len(self.counts_dict.keys())

    def calculate_scores(self, n_gram):
        precursor = n_gram[:-1]
        return (self.counts_dict.get(n_gram, 0) + self.K) / (self.precursor_dict.get(precursor, 0) + self.K * self.V)
    
    def get_log_likelihood(self, sentence):
        for i in range(self.n - 1):
            sentence = self.start_token + ' ' + sentence
        sentence += ' ' + self.end_token
        splits = sentence.split()
        scores = []
        total_score = 0
        probs = 1
        for i in range(len(splits) - (self.n - 1)):
            n_gram = ' '.join(splits[i:i+self.n])
            score = self.calculate_scores(n_gram)
            scores.append(score)
            total_score += math.log(score)
            probs *= score
        total_score /= (len(splits) - (self.n - 1))
        p_norm = math.pow(probs, 1 / (len(scores)))
        perplexity = 1 / p_norm
        return scores, total_score, perplexity


