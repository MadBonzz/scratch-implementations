from model import NGramModel, UnigramModel
from utils import combined_scores, optimize_weights

train_path = 'train.txt'
test_path = 'test.txt'
start_token = 'bos'
end_token = 'eos'
n = [1, 2, 3, 4]

with open(test_path, 'r', encoding='utf-8') as file:
    test_sentences = file.readlines()

models = []
for i in n:
    if i == 1:
        model = UnigramModel()
    else:
        model = NGramModel(i, start_token, end_token)
    model.train(train_path)
    models.append(model)

init_weights = [1 / len(models)] * len(models)
history = optimize_weights(models, 'val.txt', 8, 0.01, n_epochs=10, init_weights=init_weights)

best_key = max(history, key=lambda k: history[k]['avg_log_likelihood'])
final_weights = history[best_key]['weights']

for i in range(len(models)):
    model = models[i]
    log_scores = []
    perplexities = []
    for test_sentence in test_sentences:
        _, log_likelihood, perplexity = model.get_log_likelihood(test_sentence)
        log_scores.append(log_likelihood)
        perplexities.append(perplexity)
    print(f"For {n[i]}-gram model, the metrics are : average log likelihood -> {sum(log_scores) / len(log_scores)} and perplexity -> {sum(perplexities) / len(perplexities)}")

log_scores = []
perplexities = []
for test_sentence in test_sentences:
    _, log_likelihood, perplexity = combined_scores(models, test_sentence, init_weights)
    log_scores.append(log_likelihood)
    perplexities.append(perplexity)

print("For interpolated model with equal weights")
print(f"Average Log likelihood : {sum(log_scores) / len(log_scores)}")
print(f"Perplexity score : {sum(perplexities) / len(perplexities)}")

log_scores = []
perplexities = []
for test_sentence in test_sentences:
    _, log_likelihood, perplexity = combined_scores(models, test_sentence, final_weights)
    log_scores.append(log_likelihood)
    perplexities.append(perplexity)

print("For interpolated model with optimized weights")
print(f"Average Log likelihood : {sum(log_scores) / len(log_scores)}")
print(f"Perplexity score : {sum(perplexities) / len(perplexities)}")