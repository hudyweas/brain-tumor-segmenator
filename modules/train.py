import torch
from tqdm.autonotebook import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import gc

def train_step(model,optimizer, loss_fn,loader, device):
    epoch_loss =0.0
    model.train()

    for x,y in tqdm(loader):
        x = x.to(device,dtype = torch.float32)
        y = y.to(device, dtype = torch.float32)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()

    return epoch_loss/len(loader)

def valid_step(model,loader,loss_fn,device):
    epoch_loss = 0.0
    model.eval()

    with torch.inference_mode():
        for x,y in loader:
            x = x.to(device,dtype = torch.float32)
            y = y.to(device, dtype = torch.float32)
            pred = model(x)
            loss = loss_fn(pred,y)
            epoch_loss +=loss.item()
    return epoch_loss/len(loader)

def Train(model, optimizer, loss_fn, train_loader, valid_loader, device, saved_model_path, num_epochs = 10, patience = 3):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = train_step(model,optimizer,loss_fn,train_loader,device)
        val_loss = valid_step(model,valid_loader,loss_fn,device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), saved_model_path)
            patience_counter = 0
        else:
            patience_counter +=1
            if patience_counter >= patience:
                print("Training stopped early.")
                return train_losses, val_losses

        gc.collect()
        torch.cuda.empty_cache()
    return train_losses, val_losses

def visualizeTraining(train_losses, val_losses):
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.show(block=False)
