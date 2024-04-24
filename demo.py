import numpy as np


min = 0
max = 0
for i in range(1, 61):
    data  = np.load('D:\ProductCode\OpenFWI\FWIOpenData\FlatVel_A\data\data{:d}.npy'.format(i))
    print('max:{} min:{}'.format(np.max(data), np.min(data)))
    min = min if min < np.min(data) else np.min(data)
    max = max if max > np.max(data) else np.max(data)
print('max:{} min:{}'.format(max, min))







