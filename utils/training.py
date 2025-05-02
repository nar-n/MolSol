import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, optimizer, criterion, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch)
            loss = self.criterion(out, batch.y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        return total_loss / len(train_loader.dataset)
    
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                loss = self.criterion(out, batch.y)
                
                total_loss += loss.item() * batch.num_graphs
                y_true.append(batch.y.cpu().numpy())
                y_pred.append(out.cpu().numpy())
        
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return total_loss / len(loader.dataset), mse, r2
    
    def train(self, train_loader, val_loader, epochs=100, patience=20, model_path='best_model.pt'):
        best_val_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            val_loss, val_mse, val_r2 = self.evaluate(val_loader)
            val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(self.model.state_dict(), model_path)
                print(f'Model saved to {model_path}')
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(model_path))
        
        # Plot training curve
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_curve.png')
        
        return train_losses, val_losses
