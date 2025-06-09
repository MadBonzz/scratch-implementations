# from safetensors.torch import load_file

# tensors = load_file("model.safetensors")
# tensors = list(tensors.keys())
# tensors = [layer for layer in tensors if 'layers.0' in layer]
# print(tensors)

from safetensors.torch import load_file

tensors = load_file("model.safetensors")
q_proj_weight = tensors["model.layers.0.self_attn.k_proj.weight"]
print(q_proj_weight.shape)
print(list(tensors.keys()))