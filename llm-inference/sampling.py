import torch
import torch.nn.functional as F
import math

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
    if prev_tokens.ndim == 1:
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
    else:
        B, T = prev_tokens.shape
        if not do_multiple:
            indices = prev_tokens
            selected_logits = torch.gather(logits, 1, indices)
            positive_mask = selected_logits > 0
            positive_mask = positive_mask.to(logits.device)
            batch_indices = torch.arange(B).unsqueeze(1).expand_as(indices).to(logits.device)
            logits[batch_indices[positive_mask], indices[positive_mask]] /= penalty
            logits[batch_indices[~positive_mask], indices[~positive_mask]] *= penalty
        else:
            for i in range(B):
                for token in prev_tokens[i]:
                    if logits[i][token.item()] > 0:
                        logits[i][token.item()] /= penalty
                    else:
                        logits[i][token.item()] *= penalty
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
    
def combined_sampling(model, input_ids, penalty, include_prompt, do_multiple, temperature, k, p, max_len, eos_token_id, device='cuda'):
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    prev_tokens = None
    if include_prompt:
        prev_tokens = input_ids[0]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        probs = probs[mask]
        indices = indices[mask]
        val = torch.multinomial(probs, 1)
        id = indices[val[0].item()]
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=-1)
        prev_tokens = torch.cat((prev_tokens, id.unsqueeze(0)), dim=-1)
    else:
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        probs = probs[mask]
        indices = indices[mask]
        val = torch.multinomial(probs, 1)
        id = indices[val[0].item()]
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=-1)
        prev_tokens = id.unsqueeze(0)
    i = 1
    while((id.item() != eos_token_id) and (i < max_len)):
        logits = model(input_ids).logits[:, -1, :]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        all_false = not torch.any(mask)
        if all_false:
            id = indices[0][0]
        else:
            probs = probs[mask]
            indices = indices[mask]
            val = torch.multinomial(probs, 1)
            id = indices[val[0].item()]
        input_ids = torch.cat((input_ids, id.unsqueeze(0).unsqueeze(0)), dim=-1)
        prev_tokens = torch.cat((prev_tokens, id.unsqueeze(0)), dim=-1)
        i += 1
    return input_ids

def beam_search(model, input_ids, n_beam, penalty, include_prompt, do_multiple, temperature, k, p, max_len, eos_token_id, device='cuda'):
    beam_probs = [0] * n_beam
    model = model.to(device)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits[:, -1, :]
    prev_tokens = None
    if include_prompt:
        prev_tokens = input_ids[0]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        probs = probs[mask]
        indices = indices[mask]
        val = torch.multinomial(probs, n_beam, replacement=True)
        for i in range(len(beam_probs)):
            beam_probs[i] += math.log(probs[val][i].item())
        id = indices[val]
        input_ids = torch.stack([torch.cat([input_ids, b_i.view(1, 1)], dim=1) for b_i in id])
        prev_tokens = torch.stack([torch.cat([prev_tokens, b_i.unsqueeze(0)], dim=0) for b_i in id])
    else:
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        probs = probs[mask]
        indices = indices[mask]
        val = torch.multinomial(probs, n_beam, replacement=True)
        for i in range(len(beam_probs)):
            beam_probs[i] += math.log(probs[val][i].item())
        id = indices[val]
        input_ids = torch.stack([torch.cat([input_ids, b_i.view(1, 1)], dim=1) for b_i in id])
        prev_tokens = id.unsqueeze(0)
        prev_tokens = prev_tokens.view(n_beam, 1)
    i = 1
    input_ids = input_ids.squeeze(1)
    while(i < max_len):
        logits = model(input_ids).logits[:, -1, :]
        logits = apply_repetition_penalty(logits, prev_tokens, penalty, do_multiple)
        values, indices = torch.topk(logits, k)
        probs = apply_temperature(values, temperature)
        vals = torch.cumsum(probs, dim=-1)
        mask = vals < p
        all_false = not torch.any(mask)
        if all_false:
            id = indices[0][0]
            id = id.repeat(n_beam, 1)
            for j in range(len(beam_probs)):
                beam_probs[j] += math.log(probs[0][0].item())
        else:
            probs = probs[mask]
            indices = indices[mask]
            val = torch.multinomial(probs, n_beam, replacement=True)
            for j in range(len(beam_probs)):
                beam_probs[j] += math.log(probs[val][j].item())
            id = indices[val]
        try:
            input_ids = torch.cat((input_ids, id.unsqueeze(1)), dim=-1)
            prev_tokens = torch.cat((prev_tokens, id.unsqueeze(1)), dim=-1)
        except:
            print(input_ids.shape)
            print(prev_tokens.shape)
            print(id.shape)
        i += 1
    return beam_probs, input_ids