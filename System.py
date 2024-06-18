from Matrix import *
import numpy as np
import time

class System:
    def __init__(self, objs, forces=[], fields=[], connections=[]) -> None:
        """
        初始化系统，设置对象、力、场和连接。
        
        参数:
        objs -- 系统中的对象列表
        forces -- 每个对象所受的力列表，默认空列表
        fields -- 每个对象所在的场列表，默认空列表
        connections -- 对象间的连接列表，默认空列表
        """
        self.objs = objs
        self.data = Matrix(len(self.objs))  # 初始化对象间的连接矩阵
        self.curtime = time.time()  # 当前时间
        
        # 初始化 fields，若长度不匹配则填充 None
        if len(fields) == len(self.objs):
            self.fields = fields
        else:
            self.fields = fields + [None] * (len(self.objs) - len(fields))
        
        # 初始化 forces，若长度不匹配则填充 None
        if len(forces) == len(self.objs):
            self.forces = forces
        else:
            self.forces = forces + [None] * (len(self.objs) - len(forces))
        
        # 确保每个对象的场和力都以列表形式存储
        for i in range(len(self.objs)):
            if self.fields[i] is not None and type(self.fields[i]) != list:
                self.fields[i] = [self.fields[i]]
            if self.forces[i] is not None and type(self.forces[i]) != list:
                self.forces[i] = [self.forces[i]]

        # 设置对象间的连接
        for i in connections:
            self.data[i[0], i[1]] = i[2]
        self.kt=1
        self.dt=0
    def update(self):
        """
        更新系统状态，计算每个对象所受的力并更新其位置和速度。
        """
        t = time.time()
        dt = t - self.curtime  # 计算时间步长
        self.curtime = t
        if self.dt>0:
            dt=self.dt
        dt*=self.kt
        num_objs = len(self.objs)
        Fm = Matrix(num_objs)  # 力矩阵，用于存储对象间的力
        Fs = [np.zeros((1, 2)) for _ in range(num_objs)]  # 每个对象所受的总力

        # 计算对象间的力
        for i in range(1, num_objs):
            for j in range(i):
                if self.data[i, j] is not None:
                    # 累加所有连接产生的力
                    Fm[i, j] = sum(force.update(self.objs[i], self.objs[j]) for force in self.data[i, j])
        
        # 累加每个对象所受的力
        for i in range(num_objs):
            for j in range(num_objs):
                if Fm[i, j] is not None:
                    # 根据对象索引调整力的方向
                    Fs[i] += Fm[i, j] * (-1 if i < j else 1)
            # 考虑场对每个对象的作用力
            if self.fields[i] is not None:
                for j in range(num_objs):
                    if i == j:
                        continue
                    for k in self.fields[i]:
                        Fs[j] += k.update(self.objs[j], self.objs[i])
            # 考虑单个对象的外力
            if self.forces[i] is not None:
                for k in self.forces[i]:
                    Fs[i] += k.update(self.objs[i])
        
        # 更新每个对象的状态
        for i in range(num_objs):
            self.objs[i].update(Fs[i], dt)
        
        self.Fs = Fs  # 存储当前的力
        return Fs

    def objpos(self):
        """
        获取所有对象的位置列表。
        
        返回:
        对象位置的迭代器
        """
        return map(lambda x: x.pos, self.objs)

    def objFs(self):
        """
        获取所有对象的力列表。
        
        返回:
        对象力的迭代器
        """
        return self.Fs
