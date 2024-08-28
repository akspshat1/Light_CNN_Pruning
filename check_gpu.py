import torch # for torch only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
