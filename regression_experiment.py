import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)


def fit(inputs, targets, model, val_inputs=None, val_targets=None, patience=200, min_delta=0.001):
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Variables for early stopping
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Training loop
    for epoch in range(1500):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Validation phase
        if val_inputs is not None and val_targets is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
            
            # Early stopping logic
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                early_stop = True
                break
        
        # If no validation data is provided, stop based on training loss
        else:
            if loss < best_loss - min_delta:
                best_loss = loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                early_stop = True
                break
    
    if not early_stop:
        print("Training completed without early stopping.")


datasets = []
n_peak = 5
n_num_per_peak = 100
n_sample = n_peak * n_num_per_peak

x_grid = torch.linspace(-1,1,steps=n_sample)
x_centers = 2/n_peak * (np.arange(n_peak) - n_peak/2+0.5)
x_sample = torch.stack([torch.linspace(-1/n_peak,1/n_peak,steps=n_num_per_peak)+center for center in x_centers]).reshape(-1,)

y = 0.
for center in x_centers:
    y += torch.exp(-(x_grid-center)**2*300) 
y_sample = 0.
for center in x_centers:
    y_sample += torch.exp(-(x_sample-center)**2*300)
    
plt.plot(x_grid.detach().numpy(), y.detach().numpy())
plt.scatter(x_sample.detach().numpy(), y_sample.detach().numpy())


ys = []
for group_id in range(n_peak):
    dataset = {}
    dataset['train_input'] = x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['train_label'] = y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['test_input'] = x_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    dataset['test_label'] = y_sample[group_id*n_num_per_peak:(group_id+1)*n_num_per_peak][:,None]
    fit(dataset['train_input'], dataset['train_label'], model)
    y_pred = model(x_grid[:,None])
    ys.append(y_pred.detach().numpy()[:,0])


# Plotting
plt.subplots(1, 5, figsize=(15, 2))
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(1,6):
    plt.subplot(1,5,i)
    group_id = i - 1
    plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
    plt.plot(x_grid.detach().numpy(), ys[i-1], color='black')
    plt.xlim(-1,1)
    plt.ylim(-1,2)
