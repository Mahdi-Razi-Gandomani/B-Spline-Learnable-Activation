import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

input_size = 28 * 28
hidden_size = 256
output_size = 10
num_tasks = 3
epochs = 5
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tasks = get_permuted_mnist_tasks(num_tasks, batch_size)

# Training function and evaluate after each epoch
def train(model, train_loader, optimizer, task_idx, epoch_accuracies):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

        epoch_acc = []
        for eval_idx in range(task_idx + 1):
            acc = evaluate(model, tasks[eval_idx]['test'])
            epoch_acc.append(acc)
        epoch_accuracies.append(epoch_acc)
        print(f"Task {task_idx+1}, Epoch {epoch+1}: {epoch_acc}")

    return epoch_acc

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader.dataset)


def mnist_experiment(model):
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  epoch_accuracies = []  # Store accuracies after each epoch
  accuracies = [] # Store accuracies after each task
  for task_idx, task in enumerate(tasks):
      print(f"\nTraining on Task {task_idx + 1}")
      t = train(model, task['train'], optimizer, task_idx, epoch_accuracies)
      accuracies.append(t)
  return epoch_accuracies, accuracies

epoch_accuracies1, accuracies1 = mnist_experiment(NN(input_size, hidden_size, output_size).to(device))
epoch_accuracies2, accuracies2 = mnist_experiment(NN(input_size, hidden_size, output_size, learnable=True).to(device))
