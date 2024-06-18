import numpy as np
import misc

class Object:
    def __init__(self, m, pos, v) -> None:
        """
        初始化物体对象。

        Parameters:
        - m: 物体的质量。
        - pos: 物体的初始位置，可以是列表、元组或 numpy 数组。
        - v: 物体的初始速度，可以是列表、元组或 numpy 数组。
        """
        self.m = m
        self.pos = misc.np_ndarray(pos)  # 使用 misc 模块的 np_ndarray 函数转换为 numpy 数组
        self.v = misc.np_ndarray(v)      # 使用 misc 模块的 np_ndarray 函数转换为 numpy 数组
        self.a = misc.np_ndarray([0,0])

    def update(self, F, dt):
        """
        更新物体的位置和速度。

        Parameters:
        - F: 作用在物体上的总力，可以是 numpy 数组。
        - dt: 时间步长，用于计算位置和速度的变化。
        """
        a = F / self.m  # 计算加速度
        ka = (a - self.a)
        ba = self.a
        self.pos += (ka/6 + ba/2)*dt**2+self.v*dt  # 更新位置
        self.a=a
        self.v+=(ka/2+ba)*dt
