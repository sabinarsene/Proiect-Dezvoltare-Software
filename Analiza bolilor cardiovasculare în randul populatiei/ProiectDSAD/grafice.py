import matplotlib.pyplot as plt
from seaborn import kdeplot, scatterplot


def plot_distributie(z, y, k=0):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Distributie in axa discriminanta " + str(k + 1))
    kdeplot(x=z[:, k], hue=y, fill=True, ax=ax)
    plt.savefig("Distributie_in_axa_discriminanta_" + str(k + 1))

def scatterplot(z, zg, y, clase, k1=0, k2=1):
    fig = plt.figure(figsize=(9, 6))
    assert isinstance(fig,plt.Figure)
    ax = fig.add_subplot(1, 1, 1, aspect=1)
    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot instante si centrii in axele discriminante")
    ax.set_xlabel("z" + str(k1 + 1))
    ax.set_ylabel("z" + str(k2 + 1))
    q = len(clase)
    for i in range(q):
        x_ = z[y == clase[i], k1]
        y_ = z[y == clase[i], k2]
        ax.scatter(x_, y_, label=clase[i])
    ax.scatter(zg[:, k1], zg[:, k2], alpha=0.5, s=200, marker="s")
    ax.legend()
    plt.savefig("Plot instante si centrii in axele discriminante")

def show():
    plt.show()
