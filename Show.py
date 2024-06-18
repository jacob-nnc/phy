import pygame
from System import System
import numpy as np
import misc

class Show:
    def __init__(self,win_size,caption,center=None) -> None:
        self.window_size=np.array([win_size])

        if center is None:
            self.center=np.array([[400,300]])
        self.k=np.array([[1.,1.]])
        
        pygame.init()
        self.screen:pygame.Surface = pygame.display.set_mode(self.window_size.tolist()[0])
        
        pygame.display.set_caption(caption)
        self.transparent_surface = pygame.Surface(self.window_size.tolist()[0], pygame.SRCALPHA)
        self.transparent_surface.fill((0, 0, 0, 1))

        self.count = 0
        self.showcount = 10
        self.gridsize = np.array([[80,80]])
        self.gridstart= np.array([[0,0]])
        self.gridcanvas= pygame.Surface(self.window_size.tolist()[0], pygame.SRCALPHA)
        self.canvas = pygame.Surface(self.window_size.tolist()[0], pygame.SRCALPHA)
        self.drawgrid()
    def drawgrid(self):
        linesnum=np.ceil(self.window_size/self.gridsize).astype(int)
        self.gridcanvas.fill((0,0,0,0))
        for i in range(linesnum[0,0]):
            pygame.draw.line(
                self.gridcanvas,
                (128,128,128),
                (self.gridstart[0,0]+i*self.gridsize[0,0],0),
                (self.gridstart[0,0]+i*self.gridsize[0,0],self.window_size[0,1]),
            )
        for i in range(linesnum[0,1]):
            pygame.draw.line(
                self.gridcanvas,
                (128,128,128),
                (0,self.gridstart[0,1]+i*self.gridsize[0,1]),
                (self.window_size[0,0],self.gridstart[0,1]+i*self.gridsize[0,1]),
            )
    
    def updategrid(self,k1,k2):
        start=self.postranslate([self.gridstart],1/k1)
        start=self.postranslate(start,k2)[0]
        self.gridstart=np.mod(start,self.gridsize)
        self.gridsize=self.gridsize*k1/k2
        self.gridsize = np.where(self.gridsize < 30, self.gridsize * 2, self.gridsize)
        self.gridsize = np.where(self.gridsize > 200, self.gridsize / 2, self.gridsize)

    def updatek(self,spos):
        k=self.k.copy()
        for i in spos:
            k=np.maximum(abs((i-self.center)/(self.window_size-self.center)),k)
        k[0,0]=max(k[0,0],k[0,1])
        k[0,1]=k[0,0]
        t=self.k.copy()
        self.k=k
        return [t,k]
        
    def update(self,sys:System,msg=None):
        spos=list(sys.objpos())
        sF=list(sys.objFs())
        spos=self.postranslate(spos)
        spos=misc.np_toList(spos)
        sF=misc.np_toList(sF)

        self.count+=1

        ks=self.updatek(spos)
        self.updategrid(*ks)
        self.drawgrid()
        if self.count % self.showcount != 0:
            return
        
        if msg["pos"]==1:
            for i in spos:
                pygame.draw.circle(self.canvas,(255,255,255),i,1)
        if msg["F"]==1:
            for i in range(len(sF)):
                pygame.draw.aaline(self.canvas,(255,0,0),spos[i],[spos[i][0]+sF[i][0],spos[i][1]+sF[i][1]])
        self.screen.blit(self.canvas, (0, 0))
        self.canvas.blit(self.transparent_surface, (0, 0))
        self.screen.blit(self.gridcanvas, (0, 0))
        pygame.display.flip()
    
    def drawFunction(self,f,t):
        pos=f(t[0])
        for i in range(1,len(t)):
            pos1=f(t[i])
            pygame.draw.aaline(self.canvas,(255,0,0),pos,pos1)
            pos=pos1

    def postranslate(self,poss,k=None):
        if k is None:
            k=self.k
        return [(i-self.center)/k+self.center for i in poss]
