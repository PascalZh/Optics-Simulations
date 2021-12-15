import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10)

def f(x, a):
  return np.exp(- a * x)

for a in np.linspace(0.5, 9, 6):
  plt.plot(x, f(x, 10**(-a/10)))
plt.show()
