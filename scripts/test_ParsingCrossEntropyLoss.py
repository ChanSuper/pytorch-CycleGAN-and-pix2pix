import numpy as np

input_array = np.zeros((2, 3, 2, 2))
input_array[0, 0, :, :] = np.array([[1, 0], [1, 0]])
input_array[0, 1, :, :] = np.array([[0, 1], [0, 0]])
input_array[0, 2, :, :] = np.array([[0, 0], [0, 1]])
input_array[1, 0, :, :] = np.array([[0, 0], [0, 1]])
input_array[1, 1, :, :] = np.array([[0, 1], [1, 0]])
input_array[1, 2, :, :] = np.array([[1, 0], [0, 0]])

print(input_array)
target_array = input_array

# input_array
print('input_array: ')
input_array = input_array.transpose((1,0,2,3))
print(input_array)
c = input_array.shape[0]
n = input_array.shape[1] * input_array.shape[2] * input_array.shape[3]
print(n, c)
input_array = input_array.reshape((c, n))
print(input_array)
input_array = input_array.transpose()
print(input_array)
input_array = np.argmax(target_array, axis=1)
input_array = input_array.reshape(n)
print(input_array)

# target_array
print('target_array: ')
target_array = np.argmax(target_array, axis=1)
print(target_array)
target_array = target_array.reshape(n)
print(target_array)

print(input_array == target_array)