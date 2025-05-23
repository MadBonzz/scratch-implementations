import pandas as pd
from tqdm import tqdm
import numpy as np
import re

data_path = "C:\\Users\\shour\\OneDrive - vit.ac.in\\indic-nmt-data\\NMT  Dataset\\cleaned-test.csv"

df = pd.read_csv(data_path)
df.drop(columns=['Unnamed: 0'], inplace=True)
df1 = df[df['Lang1'].isin([
        "hin"
        ])]

df2 = df[df['Lang2'].isin([
        "hin"
    ])]

corpus = list(set(df1['Sentence_Lang1'].tolist()))
corpus2 = list(set(df2['Sentence_Lang2'].tolist()))

corpus.extend(corpus2)

def remove_special_chars_translate(input_string):
    input_string = input_string.replace('\n', '')
    input_string = re.sub(r'\d+', '', input_string)
    special_chars = "!@#$%^&*()[]{};:,./<>?\|।'`~-=+"
    translation_table = str.maketrans('', '', special_chars)
    return input_string.translate(translation_table)

for i in tqdm(range(len(corpus))):
    corpus[i] = remove_special_chars_translate(corpus[i])
    corpus[i] = corpus[i].strip()

train_idxs = np.random.randint(0, len(corpus), int(0.7 * len(corpus)))
train_sentences = [corpus[i] for i in train_idxs]
print(f"Train sentences length : {len(train_sentences)}")

non_train_sentences = list(set(corpus) - set(train_sentences))

val_idxs = np.random.randint(0, len(non_train_sentences), int(0.5 * len(non_train_sentences)))
val_sentences = [non_train_sentences[i] for i in val_idxs]
test_sentences = list(set(non_train_sentences) - set(val_sentences))

print(f"Val sentences length : {len(val_sentences)}")
print(f"Test sentences length : {len(test_sentences)}")

with open('train.txt', 'w', encoding='utf-8') as file:
    for sentence in train_sentences:
        file.write(sentence + '\n')

with open('val.txt', 'w', encoding='utf-8') as file:
    for sentence in val_sentences:
        file.write(sentence + '\n')

with open('test.txt', 'w', encoding='utf-8') as file:
    for sentence in test_sentences:
        file.write(sentence + '\n')