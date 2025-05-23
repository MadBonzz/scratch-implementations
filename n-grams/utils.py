from model import NGramModel, UnigramModel
import math

def arithmetic_mean(scores : list, weights : list):
    mean = 0
    for i in range(len(scores)):
        mean += math.log(scores[i]) * weights[i]
    return mean

def geometric_mean(scores : list, weights : list):
    product = 1
    for i in range(len(scores)):
        product *= math.pow(scores[i], weights[i])
    return product

def combined_scores(models : list[NGramModel], test_sentence, weights = None):
    if weights is None:
        weights = [1 / len(models)] * len(models)
    model_scores = []
    for model in models:
        scores, _, _ = model.get_log_likelihood(test_sentence)
        model_scores.append(scores)
    n_scores = len(model_scores[0])
    log_likelihood = 0
    probs = 1
    for i in range(n_scores):
        scores = []
        for j in range(len(model_scores)):
            scores.append(model_scores[j][i])
        log_likelihood += arithmetic_mean(scores, weights)
        probs *= geometric_mean(scores, weights)
    log_likelihood /= n_scores
    perplexity = 1 / (math.pow(probs, 1 / n_scores))
    return model_scores, log_likelihood, perplexity

def optimize_weights(models, data_path, batch_size, learning_rate, n_epochs = 10, init_weights = None):
    with open(data_path, 'r', encoding='utf-8') as file:
        sentences = file.readlines()
    if not init_weights:
        init_weights = [1 / len(models)] * len(models)
    weights = init_weights
    n_batches = len(sentences) // batch_size
    history = {}
    for epoch in range(n_epochs): 
        log_likelihoods = []
        for i in range(n_batches):
            batch = sentences[i*batch_size:i*batch_size + batch_size]
            batch_scores = [[] for _ in range(len(models))]
            for sentence in batch:
                scores, _, _ = combined_scores(models, sentence, weights)
                for i in range(len(models)):
                    batch_scores[i].extend(scores[i])
            gradients = [[] for _ in range(len(weights[1:]))]
            for i in range(len(batch_scores[0])):
                scores = [row[i] for row in batch_scores]
                p_hat = arithmetic_mean(scores, weights)
                log_likelihoods.append(p_hat)
                for j in range(len(weights[1:])):
                    gradient = batch_scores[j+1][i] - batch_scores[0][i]
                    gradient /= p_hat
                    gradients[j].append(gradient)
            for i in range(len(gradients)):
                delta = sum(gradients[i]) / len(gradients[i])
                weights[i+1] += learning_rate * delta
            weights[0] = 1 - sum(weights[1:])
        history[epoch] = {}
        history[epoch]['weights'] = weights.copy()
        history[epoch]['avg_log_likelihood'] = sum(log_likelihoods) / len(log_likelihoods)
    return history
        
