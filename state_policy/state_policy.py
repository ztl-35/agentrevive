import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import numpy as np

class StateAwarePolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3种状态: Active, Standby, Terminated
            nn.Softmax(dim=-1)
        )
        
    def forward(self, agent_memory: torch.Tensor, current_response: torch.Tensor, 
                neighbor_messages: torch.Tensor) -> torch.Tensor:
        memory_encoded, _ = self.lstm(agent_memory)
        last_memory = memory_encoded[:, -1, :]
        
        combined = torch.cat([last_memory, current_response, neighbor_messages], dim=-1)
        state_probs = self.mlp(combined)
        return state_probs
