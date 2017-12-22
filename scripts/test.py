import numpy as np

B_array_channel1 = np.array([[0,1],[1,2]])
print(B_array_channel1)
B_array_channelk = np.zeros((3, B_array_channel1.shape[0], B_array_channel1.shape[1]),
                            dtype=np.float32)
for i in range(3):
    B_array_channelk[i, :, :] = (B_array_channel1 == i).astype(np.float32)
print(B_array_channelk)