import torch

print("CUDA Available:", torch.cuda.is_available())
print("Device Count:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("GPU Name:", torch.cuda.get_device_name(0))
print("Is PyTorch Using GPU?", torch.cuda.is_initialized())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)