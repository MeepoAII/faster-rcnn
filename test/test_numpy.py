import numpy as np

# a = np.arange(0, 3)
# b = np.arange(3, 9)
# X, Y = np.meshgrid(a, b)
# print(f"X is \n {X}")
# print(f"Y is \n {Y}")
# test = np.stack((Y.ravel(), X.ravel(),
#                  Y.ravel(), X.ravel()), axis=1)
#
# print(test)


a = [[1, 2, 3], [4, 5, 6]]
b = [[7, 8, 9], [10, 11, 12]]
c = [[13, 14, 15], [16, 17, 18]]
a = np.array(a)
b = np.array(b)
c = np.array(c)

print(f"a's shape is {a.shape}")

test = np.stack((a, b, c), axis=1)
print(test)
print(test.shape)
