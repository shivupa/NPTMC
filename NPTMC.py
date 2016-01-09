import numpy as np
import scipy.constants as sc
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

#initialize
N = 125
kb = sc.k
T = sc.C2K(25)
deltaTrans = 0.15
deltaTheta = np.pi    # 15 degrees
deltaVol = 50
volFreq = 600 # Vol changed every 600
Eavg=0
P=1
numEquil = 500000
numStats = 1000000
numPrint = 10000

def rotation_matrix(angle, direction, point=None):
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float_, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M
def unit_vector(data, axis=None, out=None):
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
def ArbRot(theta,MPoint,RPoint):
    # I will rotate around 3 lines parallel to the axes that pass through OM
    # Source:http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
    #x = RPoint[0]
    #y = RPoint[1]
    #z = RPoint[2]
    a = MPoint[0]
    b = MPoint[1]
    c = MPoint[2]
    u = MPoint[0]/np.linalg.norm(MPoint)
    v = MPoint[1]/np.linalg.norm(MPoint)
    w = MPoint[2]/np.linalg.norm(MPoint)
    M = rotation_matrix(theta,[0,0,1],MPoint)
    RPoint = RPoint.transpose()
    R = M[:3, :3]
    t = M[:3, 3]
    RPoint = np.dot(R,RPoint) + t
    M = rotation_matrix(theta,[0,1,0],MPoint)
    RPoint = RPoint.transpose()
    R = M[:3, :3]
    t = M[:3, 3]
    RPoint = np.dot(R,RPoint) + t
    M = rotation_matrix(theta,[1,0,0],MPoint)
    RPoint = RPoint.transpose()
    R = M[:3, :3]
    t = M[:3, 3]
    RPoint = np.dot(R,RPoint) + t
    return RPoint
def dist(pone,ptwo):
    return np.sqrt(((ptwo[0]-pone[0])**2)+((ptwo[1]-pone[1])**2)+((ptwo[2]-pone[2])**2))
def Energy(Oloc, Mloc, H1loc, H2loc):
    #TIP4P
    A = 0.600
    C = 610
    qO = 0.0*sc.e
    qH = 0.520*sc.e
    qM = -1.040*sc.e
    E = 0
    for i in range(N):
        j = i+1
        while (j < N):

            E+= qM *qM / dist(Mloc[i],Mloc[j])
            E+= qM *qH / dist(Mloc[i],H1loc[j])
            E+= qM *qH / dist(Mloc[i],H2loc[j])
            E+= qM *qH / dist(Mloc[j],H1loc[i])
            E+= qM *qH / dist(Mloc[j],H2loc[i])
            E+= qH *qH / dist(H1loc[i],H1loc[j])
            E+= qH *qH / dist(H1loc[i],H2loc[j])
            E+= qH *qH / dist(H2loc[i],H1loc[j])
            E+= qH *qH / dist(H2loc[i],H2loc[j])
            E+= A/dist(Oloc[i],Oloc[j])
            E-= C/dist(Oloc[i],Oloc[j])
            j+=1
    return E
def plotter(step,Oloc, Mloc, H1loc, H2loc):
    Eold = Energy(Oloc, Mloc, H1loc, H2loc)
    Eavg = 0
    avgL = 0
    wat = np.random.randint(N)
    #volume move
    if (i % 600 == 0 ):
        V = VO + ((0.5*np.random.random_sample()*deltaVol)*2.0)
        Oloc *= V/VO
        Mloc *= V/VO
        H1loc *= V/VO
        H2loc *= V/VO
        Enew = Energy(Oloc, Mloc, H1loc, H2loc)
        w = (Enew - Eold)+ (P*(V-VO)) - (N*kb*T*np.log(V/VO))
        if (w<=0 or np.exp(-w/(kb*T)) > np.random.random_sample()):
            Eold=Enew
            L=np.power(V,1.0/3.0)
            Eavg +=Eold/100
            avgL +=L/100
            numaccept+=1
        else:
            Oloc /= V/VO
            Mloc /= V/VO
            H1loc /= V/VO
            H2loc /= V/VO
    else:
        oldOloc = Oloc[wat]
        oldMloc = Mloc[wat]
        oldH1loc = H1loc[wat]
        oldH2loc = H2loc[wat]
        #translate move
        move = (np.random.random_sample(3)-0.5)*deltaTrans*2.0
        Oloc[wat] += move
        Mloc[wat] += move
        H1loc[wat] += move
        H2loc[wat] += move
        #rotate move
        theta = (deltaTheta - (-deltaTheta)) * np.random.random_sample() + (-deltaTheta)
        com = (Oloc[wat]*16 +H1loc[wat] + H2loc[wat])/ 18.00
        Oloc[wat] = ArbRot(theta,com,Oloc[wat])
        H1loc[wat] = ArbRot(theta,com,H1loc[wat])
        H2loc[wat] = ArbRot(theta,com,H2loc[wat])
        Mloc[wat] = ArbRot(theta,com,Mloc[wat])

        Enew = Energy(Oloc, Mloc, H1loc, H2loc)
    if(i%numPrint==0):
        print "AVG E: ",
        print Eavg
        print "AVG L: ",
        print avgL
    if(step == 0 or i%numPrint==0):
        print step
        ax = fig.add_subplot(111, projection='3d')
        for j in range(N):
            ax.plot([Oloc[j,0],H1loc[j,0]],[Oloc[j,1],H1loc[j,1]],[Oloc[j,2],H1loc[j,2]],'k-', zdir='z')
            ax.plot([Oloc[j,0],H2loc[j,0]],[Oloc[j,1],H2loc[j,1]],[Oloc[j,2],H2loc[j,2]],'k-', zdir='z')
            #ax.plot([0,Mloc[i,0]],[0,Mloc[i,1]],[0,Mloc[i,2]],'k-', zdir='z')
            ax.scatter(Oloc[j,0],Oloc[j,1],Oloc[j,2], zdir='z', s=50, c='r', depthshade=False)
            ax.scatter(H1loc[j,0],H1loc[j,1],H1loc[j,2], zdir='z', s=20, c='k', depthshade=False)
            ax.scatter(H2loc[j,0],H2loc[j,1],H2loc[j,2], zdir='z', s=20, c='k', depthshade=False)
        return ax

L = (4.00*5) + 1.354
print "L", L
VO = L**3
index = [1.354/2.0,1.354/2.0,1.354/2.0]
Oloc = np.zeros((N,3), dtype=float)
H1loc = np.zeros((N,3), dtype=float)
H2loc = np.zeros((N,3), dtype=float)
Mloc = np.zeros((N,3), dtype=float)
OOrientationVector = np.reshape(np.tile([0,0,1],N),(N,3))
H1OrientationVector = np.reshape(np.tile([-np.sqrt(2)/2.0,-np.sqrt(2)/2.0,0],N),(N,3))
H2OrientationVector = np.reshape(np.tile([np.sqrt(2)/2.0,np.sqrt(2)/2.0,0],N),(N,3))
for i in range(5):
    for j in range(5):
        for k in range(5):
            Oloc[(i*25)+(j*5)+k] = index
            Mloc[(i*25)+(j*5)+k] = [index[0],index[1],index[2]-0.15]
            H1loc[(i*25)+(j*5)+k] = [index[0]-(1.354/2.0),index[1]-(1.354/2.0),index[2]-(1.354/2.0)]
            H2loc[(i*25)+(j*5)+k] = [index[0]+(1.354/2.0),index[1]+(1.354/2.0),index[2]-(1.354/2.0)]
            index[0] = index[0]+4.00
        index[0] = 1.354/2.0
        index[1] = index[1]+4.00
    index[0] = 1.354/2.0
    index[1] = 1.354/2.0
    index[2] = index[2]+4.00


Eold = Energy(Oloc, Mloc, H1loc, H2loc)
print dist(H1loc[0],H2loc[0]),dist(Oloc[0],H1loc[0]) ,dist(Oloc[0],H2loc[0]),dist(Oloc[0],Mloc[0])

numaccept=0


print "EQUILIBRATION START"
for i in range(10):
    wat = np.random.randint(N)
    #volume move
    if (i % 600 == 0 ):
        V = VO + ((0.5*np.random.random_sample()*deltaVol)*2.0)
        Oloc *= V/VO
        Mloc *= V/VO
        H1loc *= V/VO
        H2loc *= V/VO
        Enew = Energy(Oloc, Mloc, H1loc, H2loc)
        w = (Enew - Eold)+ (P*(V-VO)) - (N*kb*T*np.log(V/VO))
        if (w<=0 or np.exp(-w/(kb*T)) > np.random.random_sample()):
            Eold=Enew
            L=np.power(V,1.0/3.0)
            numaccept+=1
        else:
            Oloc /= V/VO
            Mloc /= V/VO
            H1loc /= V/VO
            H2loc /= V/VO

    else:
        oldOloc = Oloc[wat]
        oldMloc = Mloc[wat]
        oldH1loc = H1loc[wat]
        oldH2loc = H2loc[wat]
        #translate move
        move = (np.random.random_sample(3)-0.5)*deltaTrans*2.0
        Oloc[wat] += move
        Mloc[wat] += move
        H1loc[wat] += move
        H2loc[wat] += move
        #rotate move
        theta = (deltaTheta - (-deltaTheta)) * np.random.random_sample() + (-deltaTheta)
        com = (Oloc[wat]*16 +H1loc[wat] + H2loc[wat])/ 18.00
        Oloc[wat] = ArbRot(theta,com,Oloc[wat])
        H1loc[wat] = ArbRot(theta,com,H1loc[wat])
        H2loc[wat] = ArbRot(theta,com,H2loc[wat])
        Mloc[wat] = ArbRot(theta,com,Mloc[wat])

        Enew = Energy(Oloc, Mloc, H1loc, H2loc)
avgL = L/100
Eavg = Eold/100

print "EQUILIBRATION COMPLETE"
print "ACCEPTANCE RATIO:",numaccept/numEquil
numaccept=0
print "MAIN RUN START"
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

plt.show()
print "MAIN RUN FINISH"
print "ACCEPTANCE RATIO:",numaccept/numEquil
print "AVG E: ",
print Eavg
print "AVG L: ",
print avgL
#TODO make a check at the end to make sure distances didnt change
print dist(H1loc[0],H2loc[0]),dist(Oloc[0],H1loc[0]) ,dist(Oloc[0],H2loc[0]),dist(Oloc[0],Mloc[0])
