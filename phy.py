import sys
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
    s=System([Object(1,(0,200),(10,40))]
             ,forces=[
                 Force.Force(lambda obj:np.array([[0,obj.m*10]])-0.1*obj.v)
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
        time.sleep(0.001)

if __name__ == "__main__":
    Projectile()
