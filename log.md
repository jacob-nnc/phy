
Connection.py
```import numpy as np
import Object

class Spring:
    def __init__(self,k,L) -> None:
        self.k=k
        self.oL=L
        self.L=L

    def update(self,obj1:Object,obj2:Object):
        dP=obj1.pos-obj2.pos
        L=np.linalg.norm(dP)
        self.L=L
        if L<1e-8:
            return np.array([[0,0]])
        dP/=L
        F=np.array([[(self.L-self.oL)*self.k,0]])
        mat=np.array([[dP[0,0],dP[0,1]],[dP[0,1],-dP[0,0]]])
        return -F@mat

class Damper:
    def __init__(self,f) -> None:
        self.f=f

    def update(self,obj1:Object,obj2:Object):
        dv=obj1.v-obj2.v
        dP=obj1.pos-obj2.pos
        oL=np.linalg.norm(dP)
        dP/=oL
        dv1=dv@dP.T*dP
        F=-dv1*self.f
        return F

class Rope:
    def __init__(self,L) -> None:
        self.k=1
        self.oL=L
        self.L=L

    def update(self,obj1:Object,obj2:Object):
        dP=obj1.pos-obj2.pos
        L=np.linalg.norm(dP)
        self.L=L
        if L<1e-8:
            return np.array([[0,0]])
        dP/=L
        if self.L>self.oL:
            F=np.array([[(self.L-self.oL)*self.k,0]])
        else:
            F=np.array([[0.,0.]])
        mat=np.array([[dP[0,0],dP[0,1]],[dP[0,1],-dP[0,0]]])
        return -F@mat

class Repulsion:
    def __init__(self,k,L) -> None:
        self.k=k
        self.oL=L
        self.L=L

    def update(self,obj1:Object,obj2:Object):
        dP=obj1.pos-obj2.pos
        L=np.linalg.norm(dP)
        self.L=L
        if L<1e-8:
            return np.array([[0,0]])
        dP/=L
        if self.L<self.oL:
            F=np.array([[-self.k/(self.L-1)**4,0]])
        else:
            F=np.array([[0.,0.]])
        mat=np.array([[dP[0,0],dP[0,1]],[dP[0,1],-dP[0,0]]])
        return -F@mat

class Gravity:
    def __init__(self,G) -> None:
        self.G=G

    def update(self,obj1:Object,obj2:Object):
        dP=obj1.pos-obj2.pos
        L=np.linalg.norm(dP)
        if L<1e-8:
            return np.array([[0,0]])
        dP/=L
        F=np.array([[min(10000,obj1.m*obj2.m*self.G/L**2),0]])
        mat=np.array([[dP[0,0],dP[0,1]],[dP[0,1],-dP[0,0]]])
        return -F@mat
    ```
Field.py
```import Connection

Spring = Connection.Spring
Damper = Connection.Damper
Rope = Connection.Rope
Repulsion = Connection.Repulsion
Gravity = Connection.Gravity
```
Force.py
```import numpy as np
import misc
class Force:
    def __init__(self, F):
        if not callable(F):
            self.F=lambda _:misc.np_ndarray(F)
        else:
            self.F=F

    def update(self, obj: object):
        return self.F(obj)
```
gpthelp.py
```import os
import sys
filelist=os.listdir()
with open("log.md",'w')as f:
    for i in filelist:
        if os.path.isdir(i) or i.split('.')[-1]!='py':
            continue
        else:
            with open(i,'r') as ff:
                f.write('\n'+i+'\n')
                f.write('```'+ff.read()+'```')


```
Matrix.py
```class Matrix:
    def __init__(self,n) -> None:
        self.data=[[None]*(n-i-1) for i in range(n-1)]
    def __getitem__(self,ind):
        if ind[0]==ind[1]:
            return None
        return self.data[min(ind)][abs(ind[1]-ind[0])-1]
    
    def __setitem__(self,ind,val):
        if ind[0]==ind[1]:
            raise Exception("amns")
        self.data[min(ind)][abs(ind[1]-ind[0])-1]=val```
misc.py
```import numpy as np

def np_ndarray(a):
    if isinstance(a,np.ndarray):
        if len(a.shape)==2:
            return a
        return a.reshape(1,-1)
    else:
        if isinstance(a[0],list) or isinstance(a[0],tuple):
            return np.array(a,np.float32)
        else:
            return np.array([a],np.float32)```
Object.py
```import numpy as np
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
```
phy.py
```import sys
import os
import pygame
import numpy as np
import time
import copy

import Connection
from Object import Object
from System import System
from Show import Show
import Field
import Force

def spr_dam():
    '''
      弹簧阻尼系统，
    '''
    s = System(
        [Object(1, (350, 300), (10, 10)),
         Object(1, (500, 300), (10, -20)),
         Object(1, (300, 300), (-20, 10))],
         fields=[[Field.Damper(0.1),Field.Spring(1,100)]]*3
         )

    s.kt=5
    canvas=Show((800,600),"三体")

    msg={"pos":1,"F":1}
    while True:
        s.update()
          # 更新系统状态，获取每个对象受到的总力
        canvas.update(s,msg)

def threeStar():
    '''
      三体
    '''
    s = System(
        [Object(100, (500, 300), (0,70)),
         Object(1000, (400, 300), (0, -7)),
         Object(1,(100,300),(0,90))
         ],
         fields=[[Field.Gravity(1000)]]*3
         )

    s.kt=2
    canvas=Show((800,600),"三体")

    msg={"pos":1,"F":1}
    while True:
        s.update()
          # 更新系统状态，获取每个对象受到的总力
        canvas.update(s,msg)


def f(obj:Object):
    def warpper(t):
        return [obj.v[0,0]*t+obj.pos[0,0],obj.v[0,1]*t+5*t**2+obj.pos[0,1]]
    return warpper


def Projectile():
    s=System([Object(1,(0,200),(10,-40))]
             ,forces=[
                 Force.Force(lambda obj:np.array([[0,obj.m*10]]))
             ])
    canvas=Show((800,600),"三体")
    # ff=f(copy.deepcopy(s.objs[0]))
    msg={"pos":1,"F":1}
    s.kt=1
    while True:
        s.update()
          # 更新系统状态，获取每个对象受到的总力
        # canvas.drawFunction(ff,np.linspace(0,30,1000).tolist())
        canvas.update(s,msg)
if __name__ == "__main__":
    threeStar()
```
Show.py
```import pygame
from System import System

class Show:
    def __init__(self, win_size, caption, center=None):
        """
        初始化显示类

        参数:
        - win_size: 窗口尺寸 (宽度, 高度)
        - caption: 窗口标题
        - center: 坐标中心点，默认值为窗口中心
        """
        self.window_size = win_size
        self.center = center if center else [win_size[0] // 2, win_size[1] // 2]
        self.k = [1., 1.]

        pygame.init()
        self.canvas: pygame.Surface = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(caption)
        self.transparent_surface = pygame.Surface(self.window_size, pygame.SRCALPHA)
        self.transparent_surface.fill((0, 0, 0, 1))

        self.count = 0
        self.showcount = 10
        self.gridsize = [80, 80]
        self.gridstart = [0, 0]

    def drawgrid(self):
        """
        绘制网格
        """
        linesnum = [int(i / j) + 1 for i, j in zip(self.window_size, self.gridsize)]
        for i in range(linesnum[0]):
            pygame.draw.aaline(
                self.canvas,
                (128, 128, 128, 28),
                (self.gridstart[0] + i * self.gridsize[0], 0),
                (self.gridstart[0] + i * self.gridsize[0], self.window_size[1]),
            )
        for i in range(linesnum[1]):
            pygame.draw.aaline(
                self.canvas,
                (128, 128, 128, 28),
                (0, self.gridstart[1] + i * self.gridsize[1]),
                (self.window_size[0], self.gridstart[1] + i * self.gridsize[1]),
            )

    def updategrid(self, k1, k2):
        """
        更新网格大小和位置

        参数:
        - k1: 当前比例因子
        - k2: 更新后的比例因子
        """
        start = self.postranslate([self.gridstart], [1 / i for i in k1])
        start = self.postranslate(start, k2)[0]
        self.gridstart = [i % j for i, j in zip(self.gridstart, self.gridsize)]
        self.gridsize = [self.gridsize[0] * k1[0] / k2[0], self.gridsize[1] * k1[1] / k2[1]]

        if self.gridsize[0] < 30:
            self.gridsize[0] *= 2
        if self.gridsize[1] < 30:
            self.gridsize[1] *= 2

        if self.gridsize[0] > 200:
            self.gridsize[0] /= 2
        if self.gridsize[1] > 200:
            self.gridsize[1] /= 2

    def updatek(self, spos):
        """
        更新比例因子

        参数:
        - spos: 对象的位置列表
        """
        k = self.k[:]
        for i in spos:
            k[0] = max(abs((i[0] - self.center[0]) / (self.window_size[0] - self.center[0])), k[0])
            k[1] = max(abs((i[1] - self.center[1]) / (self.window_size[1] - self.center[1])), k[1])
        k[0] = max(k[0], k[1])
        k[1] = k[0]
        self.updategrid(self.k, k)
        self.k = k[:]

    def update(self, sys: System, msg=None):
        """
        更新并绘制系统状态

        参数:
        - sys: 系统对象
        - msg: 显示选项字典，包含是否显示位置和力
        """
        spos = list(sys.objpos())
        self.updatek(spos)
        sF = list(sys.objFs())
        spos = self.postranslate(spos)

        self.count += 1
        if self.count % self.showcount != 0:
            return
        self.canvas.blit(self.transparent_surface, (0, 0))

        self.drawgrid()
        if msg and msg.get("pos") == 1:
            for i in spos:
                pygame.draw.circle(self.canvas, (255, 255, 255), i, 1)
        if msg and msg.get("F") == 1:
            for i in range(len(sF)):
                pygame.draw.aaline(self.canvas, (255, 0, 0), spos[i], [spos[i][0] + sF[i][0], spos[i][1] + sF[i][1]])
        pygame.display.flip()

    def drawFunction(self, f, t):
        """
        绘制函数轨迹

        参数:
        - f: 函数
        - t: 时间序列
        """
        pos = f(t[0])
        for i in range(1, len(t)):
            pos1 = f(t[i])
            pygame.draw.aaline(self.canvas, (255, 0, 0), pos, pos1)
            pos = pos1

    def postranslate(self, poss, k=None):
        """
        平移坐标

        参数:
        - poss: 坐标列表
        - k: 比例因子列表，默认为当前比例因子
        """
        if k is None:
            k = self.k
        ret = [pos[:] for pos in poss]
        for i in range(len(poss)):
            ret[i][0] = (poss[i][0] - self.center[0]) / k[0] + self.center[0]
            ret[i][1] = (poss[i][1] - self.center[1]) / k[1] + self.center[1]
        return ret
```
System.py
```from Matrix import *
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
        return map(lambda x: x.pos.tolist()[0], self.objs)

    def objFs(self):
        """
        获取所有对象的力列表。
        
        返回:
        对象力的迭代器
        """
        return map(lambda x: x.tolist()[0], self.Fs)
```