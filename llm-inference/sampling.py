import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def greedy_decoding(model, input_ids, max_len, eos_token_id, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    id = torch.argmax(logits, dim=-1)
    i = 1
    while ((id.item() != eos_token_id) and (i <= max_len)):
        input_ids = torch.cat((input_ids, id.unsqueeze(0)), dim=1)
        logits = model(input_ids).logits[:, -1, :]
        id = torch.argmax(logits, dim=-1)
        i += 1
    else:
        input_ids = torch.cat((input_ids, id.unsqueeze(0)), dim=1)
        return input_ids
    
def apply_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=-1)

def temperature_sampling(model, input_ids, temp, max_len, eos_token_id, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    probs = apply_temperature(logits, temp)
    id = torch.multinomial(probs, 1)
    input_ids = torch.cat((input_ids, id), dim=-1)
    i = 1
    while ((id[0].item() != eos_token_id) and (i <= max_len)):
        logits = model(input_ids).logits[:, -1, :]
        probs = apply_temperature(logits, temp)
        id = torch.multinomial(probs, 1)
        input_ids = torch.cat((input_ids, id), dim=-1)
        i += 1
    return input_ids

def apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple):
    if not do_multiple:
        indices = prev_tokens

        selected_logits = logits[0, indices]
        positive_mask = selected_logits > 0

        logits[0, indices[positive_mask]] /= penalty
        logits[0, indices[~positive_mask]] *= penalty
    else:
        for token in prev_tokens:
            if logits[0][token.item()] > 0:
                logits[0][token.item()] /= penalty
            else:
                logits[0][token.item()] *= penalty
    return logits
    
def rp_decoding(model, input_ids, max_len, eos_token_id, penalty, include_prompt, do_multiple, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    prev_tokens = None
    if include_prompt:
        prev_tokens = input_ids[0]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        probs = F.softmax(logits, dim=1)
        id = torch.multinomial(probs, 1)
        prev_tokens = torch.cat((prev_tokens, id[0]), dim=0)
        input_ids = torch.cat((input_ids, id), dim=1)
    else:
        probs = F.softmax(logits, dim=1)
        id = torch.multinomial(probs, 1)
        prev_tokens = id[0]
        input_ids = torch.cat((input_ids, id), dim=1)
    i = 1
    while ((id[0].item() != eos_token_id) and (i <= max_len)):
        logits = model(input_ids).logits[:, -1, :]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        probs = F.softmax(logits, dim=1)
        id = torch.multinomial(probs, 1)
        prev_tokens = torch.cat((prev_tokens, id[0]), dim=0)
        input_ids = torch.cat((input_ids, id), dim=1)
        i += 1
    return input_ids

def top_k(model, input_ids, k, max_len, eos_token_id, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    val = torch.multinomial(probs, 1)
    id = indices[0][val[0].item()]
    i = 1
    while ((id.item() != eos_token_id) and (i <= max_len)):
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=1)
        logits = model(input_ids).logits[:, -1, :]
        values, indices = torch.topk(logits, k)
        probs = F.softmax(values, dim=-1)
        val = torch.multinomial(probs, 1)
        id = indices[0][val[0].item()]
        i += 1
    else:
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=1)
        return input_ids

def top_p(model, input_ids, p, max_len, eos_token_id, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    probs, indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum < p
    probs, indices = probs[mask], indices[mask]
    probs /= torch.sum(probs)
    val = torch.multinomial(probs, 1)
    id = indices[val[0].item()]
    i = 1
    while ((id.item() != eos_token_id) and (i <= max_len)):
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=1)
        logits = model(input_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        probs, indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum < p
        probs, indices = probs[mask], indices[mask]
        probs /= torch.sum(probs)
        val = torch.multinomial(probs, 1)
        id = indices[val[0].item()]
        i += 1
    else:
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=1)
        return input_ids