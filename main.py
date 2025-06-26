from model.train import train
from model.evaluate import evaluate

if __name__ == '__main__':
    print("Training the model...")
    train()
    print("Evaluating the model...")
    losses = evaluate()
    print("Evaluation complete. Sample losses:", losses[:10])