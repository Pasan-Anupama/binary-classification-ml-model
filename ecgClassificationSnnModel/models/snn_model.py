import torch
import torch.nn as nn
from .layers import LIFNeuronLayer
from .surrogate_gradient import SurrogateSpike

class SNNBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps=100):
        super(SNNBinaryClassifier, self).__init__()
        self.time_steps = time_steps
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = LIFNeuronLayer(hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, input_size)
        mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        mem2 = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)
        out_spikes = torch.zeros(x.size(0), self.fc2.out_features, device=x.device)

        for t in range(self.time_steps):
            # For simplicity, using raw input at each timestep
            cur_input = x
            h1 = self.fc1(cur_input)
            s1, mem1 = self.lif1(h1, mem1)
            h2 = self.fc2(s1)
            s2 = SurrogateSpike.apply(h2)
            mem2 += s2
            out_spikes += s2

        out = out_spikes / self.time_steps
        return out
