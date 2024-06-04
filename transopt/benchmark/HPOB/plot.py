import matplotlib.pyplot as plt
import numpy as np

# 假设你有一个N*1的数组
a = [0.62559]*1000
b = [0.31532]*50
c = [0.22537] * 20
data = np.array([a,b,c])
colors = ['red', 'green', 'blue']
# 绘制箱线图
plt.bar(x = [0.62559, 0.31532, 0.22537], height=[1000,50,20],width = 0.05, color=colors)


# 添加横纵轴标签和标题
plt.xlabel('Value')
plt.ylabel('NUmber of Value')
plt.title('Distribution of HPOBench-B Surrogate model')
plt.savefig('toy')
