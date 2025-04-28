from models import BaseRegression, LearnableRegression
from import regression_experiment

ys2 = regression_experiment(LearnableRegression().to(device), torch.optim.SGD)
ys1 = regression_experiment(BaseRegression().to(device), torch.optim.Adam)


def plot(ys):
    # Plotting
    plt.subplots(1, n_peak, figsize=(15, 2))
    plt.subplots_adjust(wspace=0.25, hspace=0)
    for i in range(1,n_peak+1):
        plt.subplot(1,n_peak,i)
        plt.plot(x_grid.detach().numpy(), y.detach().numpy(), color='black', alpha=0.1)
        plt.plot(x_grid.detach().numpy(), ys[i-1], color='black')
        plt.xlim(-1,1)
        plt.ylim(-1,2)
plot(ys2)
plot(ys1)
