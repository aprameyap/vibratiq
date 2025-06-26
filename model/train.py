import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.dataset import BearingDataset
from model.model import Autoencoder1D
from utils.logger import setup_logger
from config import CONFIG

def train():
    logger = setup_logger()
    device = CONFIG["device"]
    
    train_loader = DataLoader(BearingDataset("data/processed/train.pt"), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(BearingDataset("data/processed/val.pt"), batch_size=CONFIG["batch_size"], shuffle=False)

    model = Autoencoder1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.MSELoss()

    best_loss = float("inf")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                output = model(x)
                loss = criterion(output, x)
                val_loss += loss.item()

        logger.info(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Train Loss: {running_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            logger.info("Saved new best model.")

if __name__ == '__main__':
    train()
