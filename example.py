import torch
import torch.nn as nn

import retnet

if __name__ == "__main__":
    # verify model size for hyperparameters in paper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 400M model
    layers = 8
    hidden_dim = 2048
    ffn_size = 4096
    heads = 16

    retnet = retnet.RetNet(layers, hidden_dim, ffn_size, heads, double_v_dim=True).to(device)
    print("Model parameters:",sum(p.numel() for p in retnet.parameters() if p.requires_grad))
    print("Model layers:",layers)
