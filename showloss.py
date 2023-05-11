import glob
import matplotlib.pyplot as plt
f=glob.glob('savepath/*')
color=['r','b','g','y','k','c','m']
for i in enumerate(f):
    if(i[0]==6):
        break
    Loss=open(i[1],'r')
    losslist=Loss.readlines()
    Loss.close()
    X=[]
    Y=[]
    #print(len(losslist))
    for j in enumerate(losslist):
        if(j[0]==len(losslist)-2):
            break
        if(j[0]>=2):
            if(j[0]%1!=0):
                continue
        else:
            if (j[0] % 2 != 1):
                continue
        t=float(j[1].strip())
        #print(t)
        X.append(j[0])
        Y.append(round(t,3))
    plt.plot(X,Y,label =r"Epoch"+str(i[0]),color=color[i[0]])
    plt.legend()
plt.savefig('yaloss.png')
plt.show()
