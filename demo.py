import numpy as np
import matplotlib.pyplot as plt


# 测试data中最大值最小值
# min = 0
# max = 0
# for i in range(1, 61):
#     data  = np.load('D:\ProductCode\OpenFWI\FWIOpenData\FlatVel_A\data\data{:d}.npy'.format(i))
#     print('max:{} min:{}'.format(np.max(data), np.min(data)))
#     min = min if min < np.min(data) else np.min(data)
#     max = max if max > np.max(data) else np.max(data)
# print('max:{} min:{}'.format(max, min))

# loss可视化
# 绘制loss曲线图
x = np.arange(1, 121)
plt.figure(1)
ax11 = plt.subplot(1, 2, 1)
ax11.semilogy(x, np.log10(x), label='train')
ax11.semilogy(x, 0.1 * np.log10(x), label='test')
ax11.set_xlabel('Epochs')
ax11.set_ylabel('Loss')
ax11.set_title('MAELoss')
ax11.legend()

ax12 = plt.subplot(1, 2, 2)
ax12.plot(x, np.log10(x), label='train')
ax12.plot(x, np.log10(x), label='test')
ax12.set_yscale('log')
ax12.set_xlabel('Epochs')
ax12.set_ylabel('Loss')
ax12.set_title('MAELoss')
ax12.legend()

plt.show()




