from torch.utils.data import Dataset
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, accuracy_score
import torch.nn.functional as F

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = X_num
        self.X_cat = X_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx] 
    

class NeuralNetwork(nn.Module):
    
    def __init__(self, hidden_layers_sizes, cat_cardinalities, embedding_dims, num_numerical_features, num_target_classes):
        super().__init__()


        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_cardinality, embedding_dim)
            for cat_cardinality, embedding_dim in zip(cat_cardinalities, embedding_dims)
        ])

        self.config = {
            'hidden_layers_sizes': hidden_layers_sizes,
            'cat_cardinalities': cat_cardinalities,
            'embedding_dims': embedding_dims,
            'num_numerical_features': num_numerical_features,
            'num_target_classes': num_target_classes
        }

        total_embedding_size = sum(embedding_dims)
        input_size = total_embedding_size + num_numerical_features               #  Number of neurons in the input layer
        layers_dims = [input_size] + hidden_layers_sizes + [num_target_classes]  #  Number of neurons in each layer
        layers = []


        for i in range(len(layers_dims) - 1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))

            # Output layer without ReLU, BatchNorm and Dropout #
            if i < len(layers_dims) - 2:      
                # layers.append(nn.BatchNorm1d(layers_dims[i+1]))       # NON USARE BATCHNORM, perggiora di molto le cose          
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

        self.network = nn.Sequential(*layers)


    def _embed_input(self, x_num, x_cat):
        """Helper to process inputs"""
        embedded_cat = []
        for i, emb in enumerate(self.embeddings):
            column_embedded = emb(x_cat[:,i])
            embedded_cat.append(column_embedded)

        x_cat = torch.cat(embedded_cat, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        return x


    def forward(self, x_num, x_cat):
        x = self._embed_input(x_num, x_cat)
        return self.network(x)
    

    def fit(self, train_dataloader, valid_dataloader, device, epochs=20, loss_fn=nn.CrossEntropyLoss, lr=1e-3 ,optimizer=None, lr_scheduler=None, weights=None):

        self.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

        if weights is not None:
            train_loss_fn = loss_fn(weight=weights.to(device))  # Weighted for training
            eval_loss_fn = loss_fn()  # Unweighted for evaluation
        else:
            train_loss_fn = loss_fn()
            eval_loss_fn = loss_fn()


        best_f1, best_loss = -float('inf'), float('inf')

        for epoch in range(epochs):
            self.train()

            for x_num, x_cat, y in train_dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                
                preds = self(x_num, x_cat)
                train_loss = train_loss_fn(preds, y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                
            f1, acc, valid_loss = self.evaluate(valid_dataloader, device, eval_loss_fn)

            improved = (f1 > best_f1) or (f1 == best_f1 and valid_loss < best_loss)
            acceptable = f1 > 0.9 and valid_loss < 0.1

            if improved:
                best_f1 = f1
                best_loss = valid_loss

            if improved and acceptable:
                torch.save({'epoch': epoch,
                    'state_dict': self.state_dict(),
                    'config': self.config},
                    'best_model.pt')

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(valid_loss)
            else:
                    lr_scheduler.step() 

            print(f"--- Epoch: {epoch}  |  Loss: {valid_loss:.4f}  |  F1 Score: {f1:.4f}  |  Accuracy: {acc:.4f} ---")


    def evaluate(self, dataloader, device, loss_fn):

        self.eval()
        
        all_preds = []
        all_labels = []

        running_loss, n = 0, 0

        with torch.no_grad():
            for x_num, x_cat, y in dataloader:
                x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
                outputs = self(x_num, x_cat)
                preds = outputs.argmax(dim=1)
                batch_size = y.size(0)

                running_loss += loss_fn(outputs, y).item() * batch_size
                n += batch_size


                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        loss_val = running_loss / n
        f1  = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)

        return f1, acc, loss_val

    
    def predict(self, dataloader, device):
        self.eval()
        all_preds = []

        with torch.inference_mode():
            for x_num, x_cat, _ in dataloader:  
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                outputs = self(x_num, x_cat)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds)


    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config 
        }, path)


    @classmethod
    def load(cls, path, device='cuda'):
        if not torch.cuda.is_available():
            print("[WARNING] CUDA non disponibile. Caricamento su CPU.")
            device = 'cpu'

        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model