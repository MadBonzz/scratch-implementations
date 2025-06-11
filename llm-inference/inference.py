from transformers import AutoTokenizer, AutoModelForCausalLM
import sampling as sm

device = 'cuda'
model_id = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
text = "Hello I am Shakespeare"
inputs = tokenizer.encode(text, return_tensors="pt").to(device)

out = sm.top_p(model, inputs, 0.9, 20, 0, device)
print(tokenizer.decode(out[0]))
out = sm.top_p(model, inputs, 0.6, 20, 0, device)
print(tokenizer.decode(out[0]))