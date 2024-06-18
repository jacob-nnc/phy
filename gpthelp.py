import os
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


