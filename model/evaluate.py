import torch
from torch.utils.data import DataLoader
from model.dataset import BearingDataset
from model.model import Autoencoder1D
from config import CONFIG

def evaluate():
    test_loader = DataLoader(BearingDataset("data/processed/test.pt"), batch_size=CONFIG["batch_size"], shuffle=False)
    device = CONFIG["device"]

    model = Autoencoder1D().to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()

    criterion = torch.nn.MSELoss(reduction='none')
    all_losses = []
    
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            output = model(x)
            loss = criterion(output, x)
            loss = loss.mean(dim=(1, 2))  # batch-wise loss
            all_losses.extend(loss.cpu().numpy())

    return all_losses

if __name__ == '__main__':
    losses = evaluate()
    print("Test MSE per sample:", losses)