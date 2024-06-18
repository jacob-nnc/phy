import numpy as np
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
    