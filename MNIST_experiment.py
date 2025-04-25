import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Settings
num_tasks = 6
epochs = 5
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
tasks = get_permuted_mnist_tasks(num_tasks, batch_size)


# Training function
def train(model, train_loader, optimizer):
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

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


# Main loop
accuracies = []
for task_idx, task in enumerate(tasks):
    print(f"Training on Task {task_idx + 1}")
    train(model, task['train'], optimizer)

    task_acc = []
    for eval_idx in range(task_idx + 1):
        acc = evaluate(model, tasks[eval_idx]['test'])
        task_acc.append(acc)
    accuracies.append(task_acc)
    print(f"Accuracies after Task {task_idx + 1}: {task_acc}\n")

# Final summary
print("Final accuracy matrix:")
for i, row in enumerate(accuracies):
    print(f"Task {i+1}: {['{:.2f}'.format(a*100) for a in row]}")


# Plot forgetting
for task_id in range(num_tasks):
    accs = [accuracies[i][task_id] if task_id < len(accuracies[i]) else None for i in range(num_tasks)]
    plt.plot(range(1, num_tasks+1), accs, marker='o', label=f"Task {task_id+1}")

plt.title("Catastrophic Forgetting: Accuracy vs. Task")
plt.xlabel("Task Number")
plt.ylabel("Accuracy")
plt.yticks([0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
plt.xticks(range(1, num_tasks + 1))
plt.grid(False)
plt.tight_layout()
plt.show()

#plot average accuracy across all tasks
avg_accuracies = [np.mean(acc) for acc in accuracies]
plt.plot(range(1, num_tasks+1), avg_accuracies, marker='s', linestyle='-', linewidth=3, color='k', label='Average Accuracy')

plt.title("Accuracy vs. Task (Individual Tasks & Average)")
plt.xlabel("Task Number")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
