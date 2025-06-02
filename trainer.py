"""
Model trainer for creating and training the digital twin.
"""
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

class CodeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """Initialize the code dataset.
        
        Args:
            data_path: Path to the collected code data
            tokenizer: Tokenizer for processing code
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load and preprocess the code data."""
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            
        processed_data = []
        for item in raw_data:
            if 'code' in item and 'style' in item:
                processed_data.append({
                    'code': item['code'],
                    'style': item['style'],
                    'patterns': item.get('patterns', {})
                })
        return processed_data
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize code
        code_tokens = self.tokenizer(
            item['code'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Convert style and patterns to tensor
        style_tensor = torch.tensor(list(item['style'].values()), dtype=torch.float)
        patterns_tensor = torch.tensor(list(item['patterns'].values()), dtype=torch.float)
        
        return {
            'input_ids': code_tokens['input_ids'].squeeze(),
            'attention_mask': code_tokens['attention_mask'].squeeze(),
            'style': style_tensor,
            'patterns': patterns_tensor
        }

class DigitalTwinModel(nn.Module):
    def __init__(self, base_model_name: str, style_dim: int, pattern_dim: int):
        """Initialize the digital twin model.
        
        Args:
            base_model_name: Name of the base language model
            style_dim: Dimension of style features
            pattern_dim: Dimension of pattern features
        """
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Style and pattern encoders
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.base_model.config.hidden_size)
        )
        
        self.pattern_encoder = nn.Sequential(
            nn.Linear(pattern_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.base_model.config.hidden_size)
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.base_model.config.hidden_size * 3, self.base_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                style: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            style: Style features
            patterns: Pattern features
            
        Returns:
            Model output
        """
        # Get base model outputs
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Encode style and patterns
        style_encoded = self.style_encoder(style)
        pattern_encoded = self.pattern_encoder(patterns)
        
        # Combine features
        combined = torch.cat([
            base_outputs.hidden_states[-1],
            style_encoded.unsqueeze(1).expand(-1, base_outputs.hidden_states[-1].size(1), -1),
            pattern_encoded.unsqueeze(1).expand(-1, base_outputs.hidden_states[-1].size(1), -1)
        ], dim=-1)
        
        # Fuse features
        fused = self.fusion_layer(combined)
        
        return fused

class DigitalTwinTrainer:
    def __init__(self, model: DigitalTwinModel, learning_rate: float = 1e-5):
        """Initialize the trainer.
        
        Args:
            model: Digital twin model
            learning_rate: Learning rate for training
        """
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader: DataLoader, num_epochs: int,
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> List[float]:
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            num_epochs: Number of training epochs
            device: Device to train on
            
        Returns:
            List of training losses
        """
        self.model.to(device)
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                style = batch['style'].to(device)
                patterns = batch['patterns'].to(device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, style, patterns)
                
                # Calculate loss
                loss = self.criterion(outputs, input_ids)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_epoch_loss = epoch_loss / len(train_loader)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
            
        return losses
        
    def save_model(self, path: str):
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_digital_twin(data_path: str, output_path: str, config: Dict):
    """Train a digital twin model.
    
    Args:
        data_path: Path to the collected data
        output_path: Path to save the trained model
        config: Training configuration
    """
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    model = DigitalTwinModel(
        base_model_name=config['base_model'],
        style_dim=config['style_dim'],
        pattern_dim=config['pattern_dim']
    )
    
    # Create dataset and dataloader
    dataset = CodeDataset(data_path, tokenizer, config['max_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # Initialize trainer
    trainer = DigitalTwinTrainer(model, learning_rate=config['learning_rate'])
    
    # Train model
    losses = trainer.train(
        train_loader=dataloader,
        num_epochs=config['num_epochs'],
        device=config['device']
    )
    
    # Save model
    trainer.save_model(output_path)
    
    return losses 