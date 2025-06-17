from transformers import AutoTokenizer, AutoModelForCausalLM
import sampling as sm

device = 'cuda'
model_id = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
text = "Hello I am Shakespeare"
inputs = tokenizer.encode(text, return_tensors="pt").to(device)

print("With multiple as false")
probs, out = sm.beam_search(model, inputs, 2, 1.3, True, False, 0.8, 40, 0.9, 40, 0, 'cuda')
print(tokenizer.decode(out[probs.index(max(probs))]))
print("With multiple as true")
probs, out = sm.beam_search(model, inputs, 2, 1.3, False, True, 0.8, 40, 0.9, 40, 0, 'cuda')
print(tokenizer.decode(out[probs.index(max(probs))]))
