from safetensors.torch import load_file
from safetensors.torch import load_model
from model import SmolLM

tensors = load_file("model.safetensors")
# q_proj_weight = tensors["model.layers.0.self_attn.k_proj.weight"]
# print(q_proj_weight.shape)
print(list(tensors.keys()))

model = SmolLM(49152, 576, 30, 9, 3, 64, 1536, 2048)
#load_model(model, "model.safetensors")

for key in model.state_dict().keys():
    print(key)
