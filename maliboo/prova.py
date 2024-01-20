import numpy as np

a = np.ones(4)
print(a.shape)
b = np.array([1,1,1,1]).reshape((4,1))
print(b.shape)
b = b.reshape(-1)
print(b.shape)
b = b.reshape(-1)
print(b.shape)

c = a.copy()
print(c.shape)


a = 1e-19
b = np.log(a) + 1/(2*a**2)
print(b)

print(np.isnan(np.log(-2)))