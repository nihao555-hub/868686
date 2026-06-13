# -*- coding: utf-8 -*-
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import numpy as np
import matplotlib.pyplot as plt
EuclDist = scipy.spatial.distance.euclidean
def gap(data, resf=None, nrefs=10, ks=range(1,10)):
    #计算间隔统计量
    shape = data.shape
    if resf == None:
        x_max = data.max(axis=0)
        x_min = data.min(axis=0)
        dists = np.matrix(np.diag(x_max-x_min))
        rands = np.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+x_min
    else:
        rands = refs
    gaps = np.zeros((len(ks),))
    gapDiff = np.zeros(len(ks)-1,)
    sdk = np.zeros(len(ks),)
    for (i,k) in enumerate(ks):
        (cluster_mean, cluster_res) = scipy.cluster.vq.kmeans2(data, k)
        Wk = sum([EuclDist(data[m,:], cluster_mean[cluster_res[m],:]) for m in range(shape[0])])
        WkRef = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            WkRef[j] = sum([EuclDist(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
        gaps[i] = np.log(np.mean(WkRef))-np.log(Wk)
        sdk[i] = np.sqrt((1.0+nrefs)/nrefs)*np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i-1] = gaps[i-1] - gaps[i] + sdk[i]
    return gaps, gapDiff

mean = (1, 2)
cov = [[1, 0], [0, 1]]
Nf = 1000;
dat1 = np.zeros((3000,2))
dat1[0:1000,:] = np.random.multivariate_normal(mean, cov, 1000)
mean = [5, 6]
dat1[1000:2000,:] = np.random.multivariate_normal(mean, cov, 1000)
mean = [3, -7]
dat1[2000:3000,:] = np.random.multivariate_normal(mean, cov, 1000)
plt.plot(dat1[::,0], dat1[::,1], 'b.', linewidth=1)

gaps,gapsDiff = gap(dat1)
plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots(1,1)
plt.plot(gaps, 'g-o')
plt.xlabel('Number of clusters K')
plt.ylabel('gaps')

plt.subplots(1,1)
plt.bar(np.arange(len(gapsDiff)),gapsDiff)
plt.xlabel('Number of clusters K')
plt.ylabel('Gap(k)-Gap(k+1)+s_k+1 ')

plt.show()
