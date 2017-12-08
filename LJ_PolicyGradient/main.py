import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from LJEnvironment import *
from doubleLJ import *
from fingerprintFeature import *
from plot_structure_and_feature import *

r0 = 1.0
eps = 0
sigma= 1
params = (r0,eps,sigma)

LJEnv = LJEnvironment(params)

N = 4

xlist1 = np.array([0,0,-1,-1,1,0,1])
ylist1 = np.array([0,1,1,0,0,-1,-1])


XY = np.array([LJEnv.gridToXY(np.array([xlist1[i],ylist1[i]])) for i in list(range(len(xlist1)))])
xlist1 = XY.T[0]
ylist1 = XY.T[1]

X1 = np.zeros(2*N)
for i in range(N):
    X1[2*i] = xlist1[i]
    X1[2*i+1] = ylist1[i]
    

fpf = fingerprintFeature()
feat1 = fpf.get_singleFeature(X1)




xlist2 = np.array([-1,0,-2,-2,1,1,1])
ylist2 = np.array([2,1,2,1,0,-2,-1])

XY = np.array([LJEnv.gridToXY(np.array([xlist2[i],ylist2[i]])) for i in list(range(len(xlist2)))])
xlist2 = XY.T[0]
ylist2 = XY.T[1]

X2 = np.zeros(2*N)
for i in range(N):
    X2[2*i] = xlist2[i]
    X2[2*i+1] = ylist2[i]

feat2 = fpf.get_singleFeature(X2)






xlist3 = np.array([0,0,2,-2,1,1,1])
ylist3 = np.array([0,1,3,3,-1,-2,-1])


XY = np.array([LJEnv.gridToXY(np.array([xlist3[i],ylist3[i]])) for i in list(range(len(xlist3)))])
xlist3 = XY.T[0]
ylist3 = XY.T[1]

X3 = np.zeros(2*N)
for i in range(N):
    X3[2*i] = xlist3[i]
    X3[2*i+1] = ylist3[i]

feat3 = fpf.get_singleFeature(X3)





# plt.plot(a)

fig = plt.figure(figsize=(8,10))
ax_struct1 = fig.add_subplot(321)
ax_struct1.set_xlim([-5,5])
ax_struct1.set_ylim([-5,5])
ax_struct1.set_title('Structure')
ax_struct1.set_xlabel('x')
ax_struct1.set_ylabel('y')
ax_struct1.plot(xlist1,ylist1,'ro')

ax_feat1 = fig.add_subplot(322)
ax_feat1.set_title('Fingerprint feature')
# ax_struct1.set_xlabel('')
# ax_struct1.set_ylabel('y')
ax_feat1.plot(feat1,'k')



ax_struct2 = fig.add_subplot(323)
ax_struct2.set_xlim([-5,5])
ax_struct2.set_ylim([-5,5])
ax_struct2.set_title('Structure')
ax_struct2.set_xlabel('x')
ax_struct2.set_ylabel('y')
ax_struct2.plot(xlist2,ylist2,'bo')


ax_feat2 = fig.add_subplot(324)
ax_feat2.set_title('Fingerprint feature')
ax_feat2.plot(feat2,'k')





ax_struct3 = fig.add_subplot(325)
ax_struct3.set_xlim([-5,5])
ax_struct3.set_ylim([-5,5])
ax_struct3.set_title('Structure')
ax_struct3.set_xlabel('x')
ax_struct3.set_ylabel('y')
ax_struct3.plot(xlist3,ylist3,'go')


ax_feat3 = fig.add_subplot(326)
ax_feat3.set_title('Fingerprint feature')
ax_feat3.plot(feat3,'k')












fig.tight_layout()
fig.savefig('structFeatFig',dpi=200)

