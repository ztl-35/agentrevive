import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio

from node import Node
from agents.agent_registry import AgentRegistry
import random

class AgentReviveGraph:
    def __init__(self, 
                 domain: str,
                 llm_name: str,
                 agent_names: List[str],
                 decision_method: str,
                 num_rounds: int = 3,
                 risk_coefficient: float = 0.5,
                 survival_threshold: float = 0.6):
        
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.num_rounds = num_rounds
        self.risk_coefficient = risk_coefficient
        self.survival_threshold = survival_threshold
        
        self.nodes = self._init_nodes(agent_names)
        self.decision_node = self._init_decision_node(decision_method)
        
        self.state_policy = StateAwarePolicy(input_dim=512)  # 假设的输入维度
        
        self.agent_states = {agent_id: "Active" for agent_id in self.nodes.keys()}
        self.agent_memories = {agent_id: [] for agent_id in self.nodes.keys()}
        self.agent_responses = {agent_id: None for agent_id in self.nodes.keys()}
        
        self.spatial_logits = nn.Parameter(
            torch.randn(len(agent_names), len(agent_names)) * 0.1
        )
        self.temporal_logits = nn.Parameter(
            torch.randn(num_rounds-1, len(agent_names), len(agent_names)) * 0.1
        )
    
    def _init_nodes(self, agent_names: List[str]) -> Dict[str, Any]:
        nodes = {}
        for i, name in enumerate(agent_names):
            nodes[f"agent_{i}"] = {
                'name': name,
                'role': self._get_agent_role(name),
                'state': 'Active',
                'memory': [],
                'responses': []
            }
        return nodes
    
    def _init_decision_node(self, decision_method: str) -> Any:
        
        return {"type": "decision", "method": decision_method}
    
    def _get_agent_role(self, agent_name: str) -> str:
        role_mapping = {
            'MathSolver': 'Math Expert',
            'Historian': 'History Expert', 
            'Critic': 'Critor',
            'Programmer': 'Programmer',
            'KnowledgeExpert': 'Knowledge Expert'
        }
        return role_mapping.get(agent_name, 'General Expert')
    
    def state_aware_message_passing(self, round_idx: int) -> Dict[str, torch.Tensor]:
        
        messages = {}
        
        for agent_id, agent in self.nodes.items():
            if self.agent_states[agent_id] in ["Active", "Standby"]:
                spatial_messages = self._aggregate_spatial_messages(agent_id, round_idx)
                
                temporal_messages = self._aggregate_temporal_messages(agent_id, round_idx)
                
                combined_messages = torch.cat([spatial_messages, temporal_messages])
                messages[agent_id] = combined_messages
            else:
                messages[agent_id] = torch.zeros(256)  # 零向量
        
        return messages
    
    def _aggregate_spatial_messages(self, agent_id: str, round_idx: int) -> torch.Tensor:
        spatial_neighbors = self._get_spatial_neighbors(agent_id)
        messages = []
        
        for neighbor_id in spatial_neighbors:
            if (self.agent_states[neighbor_id] in ["Active", "Standby"] and 
                self.agent_responses[neighbor_id] is not None):
                
                weight = torch.sigmoid(self.spatial_logits[
                    list(self.nodes.keys()).index(agent_id),
                    list(self.nodes.keys()).index(neighbor_id)
                ])
                
                message_vector = torch.randn(128) * weight
                messages.append(message_vector)
        
        return torch.mean(torch.stack(messages), dim=0) if messages else torch.zeros(128)
    
    def _aggregate_temporal_messages(self, agent_id: str, round_idx: int) -> torch.Tensor:
        if round_idx == 0:
            return torch.zeros(128)
            
        temporal_neighbors = self._get_temporal_neighbors(agent_id)
        messages = []
        
        for neighbor_id in temporal_neighbors:
            if (self.agent_states[neighbor_id] in ["Active", "Standby"] and
                self.agent_responses[neighbor_id] is not None):
                
                weight = torch.sigmoid(self.temporal_logits[
                    round_idx-1,
                    list(self.nodes.keys()).index(agent_id),
                    list(self.nodes.keys()).index(neighbor_id)
                ])
                
                message_vector = torch.randn(128) * weight
                messages.append(message_vector)
        
        return torch.mean(torch.stack(messages), dim=0) if messages else torch.zeros(128)
    
    def _get_spatial_neighbors(self, agent_id: str) -> List[str]:
        
        return [id for id in self.nodes.keys() if id != agent_id]
    
    def _get_temporal_neighbors(self, agent_id: str) -> List[str]:
        return list(self.nodes.keys())
    
    def compute_risk_estimator(self, round_idx: int) -> torch.Tensor:
        
        active_agents = [agent_id for agent_id, state in self.agent_states.items() 
                        if state == "Active"]
        
        if not active_agents:
            return torch.tensor(0.0)
        
        active_messages = []
        for agent_id in active_agents:
            if self.agent_responses[agent_id] is not None:
                message_vec = torch.randn(256)  # 实际应该使用编码器
                active_messages.append(message_vec)
        
        if not active_messages:
            return torch.tensor(0.0)
        
        avg_message = torch.mean(torch.stack(active_messages), dim=0)
        
        kl_divergences = []
        for message in active_messages:
            kl = F.kl_div(
                F.log_softmax(message, dim=0),
                F.softmax(avg_message, dim=0),
                reduction='batchmean'
            )
            kl_divergences.append(kl)
        
        risk = -torch.mean(torch.stack(kl_divergences))
        return risk
    
    def state_transition(self, round_idx: int, messages: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        log_probs = []
        
        for agent_id, agent in self.nodes.items():
            if self.agent_states[agent_id] == "Terminated":
                continue
            
            memory_tensor = self._encode_memory(agent_id)
            response_tensor = self._encode_response(agent_id)
            message_tensor = messages[agent_id]
            
            state_probs = self.state_policy(memory_tensor, response_tensor, message_tensor)
            
            next_state_idx = torch.multinomial(state_probs, 1).item()
            next_state = ["Active", "Standby", "Terminated"][next_state_idx]
            
            log_prob = torch.log(state_probs[0, next_state_idx])
            log_probs.append(log_prob)
            
            self.agent_states[agent_id] = next_state
        
        return torch.sum(torch.stack(log_probs)) if log_probs else torch.tensor(0.0)
    
    def _encode_memory(self, agent_id: str) -> torch.Tensor:
        memory = self.agent_memories[agent_id]
        if not memory:
            return torch.zeros(1, 1, 256)  # 空记忆
        
        encoded_memory = torch.randn(1, len(memory), 256)
        return encoded_memory
    
    def _encode_response(self, agent_id: str) -> torch.Tensor:
        response = self.agent_responses[agent_id]
        if response is None:
            return torch.zeros(128)
        
        return torch.randn(128)
    
    def execute_agent(self, agent_id: str, input_data: Dict[str, Any], round_idx: int) -> str:
        
        agent_state = self.agent_states[agent_id]
        
        if agent_state == "Active":
            response = self._generate_new_response(agent_id, input_data, round_idx)
            self.agent_responses[agent_id] = response
            
        elif agent_state == "Standby":
            if self.agent_responses[agent_id] is not None:
                response = self._compress_historical_response(agent_id)
            else:
                response = "No previous response available."
                
        else:  # Terminated
            response = "Agent terminated."
        
        # 更新记忆
        self.agent_memories[agent_id].append(response)
        
        return response
    
    def _generate_new_response(self, agent_id: str, input_data: Dict[str, Any], 
                             round_idx: int) -> str:
        agent = self.nodes[agent_id]
        return f"{agent['role']} response for round {round_idx}"
    
    def _compress_historical_response(self, agent_id: str) -> str:
        historical_response = self.agent_responses[agent_id]
        return f"Summarized: {historical_response}"
    
    def state_aware_edge_optimization(self) -> torch.Tensor:

        survival_rates = {}
        for agent_id in self.nodes.keys():
            non_terminated_count = sum(1 for state in self.agent_states.values() 
                                     if state != "Terminated")
            survival_rates[agent_id] = non_terminated_count / len(self.nodes)
        
        node_mask = {}
        for agent_id, rate in survival_rates.items():
            node_mask[agent_id] = 1 if rate >= self.survival_threshold else 0
        
        effective_spatial_adj = self._apply_node_mask_to_adjacency(
            self.spatial_logits, node_mask
        )
        effective_temporal_adj = self._apply_node_mask_to_adjacency(
            self.temporal_logits, node_mask
        )
        
        sparsity_loss = self._compute_sparsity_loss(effective_spatial_adj, effective_temporal_adj)
        
        return sparsity_loss
    
    def _apply_node_mask_to_adjacency(self, adjacency: torch.Tensor, 
                                    node_mask: Dict[str, int]) -> torch.Tensor:
        mask_vector = torch.tensor([node_mask[agent_id] for agent_id in self.nodes.keys()])
        mask_matrix = torch.outer(mask_vector, mask_vector)
        
        if len(adjacency.shape) == 3:  # 时间邻接矩阵
            masked_adj = adjacency * mask_matrix.unsqueeze(0)
        else:  # 空间邻接矩阵
            masked_adj = adjacency * mask_matrix
        
        return masked_adj
    
    def _compute_sparsity_loss(self, spatial_adj: torch.Tensor, 
                             temporal_adj: torch.Tensor) -> torch.Tensor:
        spatial_loss = torch.norm(spatial_adj, p='nuc')
        
        temporal_loss = torch.tensor(0.0)
        if len(temporal_adj.shape) == 3:
            for t in range(temporal_adj.shape[0]):
                temporal_loss += torch.norm(temporal_adj[t], p='nuc')
        
        return spatial_loss + temporal_loss
    
    async def arun(self, input_data: Dict[str, Any], num_rounds: int = None) -> List[str]:
        
        if num_rounds is None:
            num_rounds = self.num_rounds
        
        all_answers = []
        trajectory_reward = torch.tensor(0.0)
        
        for round_idx in range(num_rounds):
            messages = self.state_aware_message_passing(round_idx)
            
            state_log_prob = self.state_transition(round_idx, messages)
            trajectory_reward = trajectory_reward + state_log_prob
            
            round_answers = {}
            for agent_id in self.nodes.keys():
                response = await self._async_execute_agent(agent_id, input_data, round_idx)
                round_answers[agent_id] = response
            
            all_answers.append(round_answers)
            
            risk = self.compute_risk_estimator(round_idx)
            trajectory_reward = trajectory_reward + self.risk_coefficient * risk
            
            edge_loss = self.state_aware_edge_optimization()
            trajectory_reward = trajectory_reward - edge_loss
        
        # 最终决策
        final_answer = self._make_final_decision(all_answers)
        
        return [final_answer], trajectory_reward
    
    async def _async_execute_agent(self, agent_id: str, input_data: Dict[str, Any], 
                                 round_idx: int) -> str:
        """异步执行智能体"""
        return self.execute_agent(agent_id, input_data, round_idx)
    
    def _make_final_decision(self, all_answers: List[Dict[str, str]]) -> str:
        """生成最终决策"""
        for agent_id, state in self.agent_states.items():
            if state == "Active" and all_answers:
                return all_answers[-1].get(agent_id, "No answer")
        return "No active agents found"
