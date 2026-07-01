import torch
print("CUDA:", torch.cuda.is_available())
print("VRAM:", round(torch.cuda.get_device_properties(0).total_mem/1e9,1), "GB")
