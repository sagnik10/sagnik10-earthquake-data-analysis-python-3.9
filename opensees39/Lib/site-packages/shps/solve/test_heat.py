
#!/usr/bin/env python
# Claudio Perez
# Fall 2022

import numpy as np
from numpy import sin, sinh, pi
import matplotlib.pyplot as plt

def u(x,y,N):
    return sum(800.0/(sinh(n*pi)*(pi*n)**3)*sin(n*pi*x/10)*sinh(n*pi*(10-y)/10)
            for n in range(1,N,2))

def plot_analytic(ax):
    X,Y = np.meshgrid(*[np.linspace(0, 10, 200)]*2)

    # fig, ax = plt.subplots()
    CS = ax.contour(X, Y, u(X,Y,200))
    ax.clabel(CS, inline=True, fontsize=10)
    # ax.set_title("Analytic Solution")
    # plt.show()
    return ax

if __name__=="__main__":
    plot_analytic(plt.figure().add_subplot(projection="3d"))
    plt.show()


