import numpy as np
import matplotlib.pyplot as plt

# number of points
n = 50

# length of road
length = 10

# distance between points
dx = length/(n-1)

# heat diffusion coefficient
alpha = 0.1

# time step
dt = 0.01

# total simulation steps
steps = 5000

# x position
x = np.linspace(0,length,n)

# initial temperature
u = np.zeros(n)

# make the middle part hot
u[20:30] = 100

# save the initial state
initial_u = u.copy()

for step in range(steps):

    new_u = u.copy()

    for i in range(1,n-1):

        new_u[i] = (u[i] + alpha * dt * (u[i+1] - 2 * u[i] + u[i-1])/dx ** 2)

    u = new_u

plt.plot(x,initial_u,label = "Initial")
plt.plot(x,u,label = "After simulation")

plt.xlabel("Position")
plt.ylabel("Temperature")
plt.legend()

plt.show()


#一维热传导方程，用来观测中间很热时，热量如何像两边扩散 也就是∂t/∂u​=α(∂^2u/∂x^2)​这个方程
#设置我们的物理模型是高度10，长度分成50个位置，因此初始状态是如图所示蓝色部分，即中间高，两边低，那么在现实生活中，中间的高温回慢慢的像两侧扩散也就是图中黄线所表示的含义
#代码中的重要参数：
    # alpha指的是扩散的有多快
    # dt计算每一次推进多少时间，会一点一点跳，而不是直接到5或者10秒之后
    # steps = 500 指的是我们总共模拟500个steps
    # u[20:30] = 100 我们把20-30之间设置成100度
# new_u[i]的部分就是整个PDE被转换成python代码的地方，也就是∂t/∂u​=α(∂^2u/∂x^2)

# 后续和上周找到的github应该可以联动进行测试，首先测试jupedsim