class Matrix:
    def __init__(self,n) -> None:
        self.data=[[None]*(n-i-1) for i in range(n-1)]
    def __getitem__(self,ind):
        if ind[0]==ind[1]:
            return None
        return self.data[min(ind)][abs(ind[1]-ind[0])-1]
    
    def __setitem__(self,ind,val):
        if ind[0]==ind[1]:
            raise Exception("amns")
        self.data[min(ind)][abs(ind[1]-ind[0])-1]=val