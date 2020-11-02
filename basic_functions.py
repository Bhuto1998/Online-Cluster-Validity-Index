import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Function-1: Finding the closest cluster of a point given the point and centroids
def find_closest_cluster(x, centroids):
    closest_cluster, closest_distance = 999999, 999999 # initially invalid
    K = len(centroids)
    for k in range(K): # no. clusters
        temp = 0
        for j in range(len(centroids[k])):
            temp = temp + (x[j]-centroids[k][j])**2
    distance = temp # Euclidean distance
    if distance < closest_distance:
      closest_cluster  = k
      closest_distance = distance
    return closest_cluster

#Function-2: Sequential K-means
def seqKmean(k,X):
    if len(X)<k:
        print("length is an issue")
    else:
        centroids = X[:k]
        counter = np.ones(k)
        for x_n in X[k:]:
            closest_k = find_closest_cluster(x_n,centroids)
            counter[closest_k] +=1
            temp = np.subtract(x_n,centroids[closest_k])
            temp = np.divide(temp,counter[closest_k])
            centroids[closest_k] = np.add(centroids[closest_k],temp)
        return centroids , counter

#Function-3: Calculating Euclidean Norm
def norm(v):
    temp = 0
    for i in range(len(v)):
        temp = temp + (v[i]**2)
    temp = temp**0.5
    return temp

#Function-4: Exponentially fading E_K
def eK(K,X,l):
    if len(X)<K:
        print("length is an issue")
    else:
        centroids = X[:K]
        counter = np.ones(K)
        p = np.zeros(K)
        temp_len = len(X) - K
        eK = np.zeros(temp_len)
        temp_len = (K,len(X[K]))
        g = np.zeros(temp_len)
        
        for j in range(K,len(X),1):      
            closest_K = find_closest_cluster(X[j],centroids)
            m = counter.copy()
            counter[closest_K] +=1
            
            temp3 = np.subtract(counter,m)
            temp_centroids = centroids.copy()
            temp = np.subtract(X[j],temp_centroids[closest_K])
            temp = np.multiply(temp, l)
            temp_centroids[closest_K] = np.add(temp_centroids[closest_K],temp)
            #update stage
            for i in range(K):
                temp1 = np.subtract(centroids[i],temp_centroids[i])
                temp1 = np.transpose(temp1)
                q = np.matmul(temp1,g[i])
                b = norm(np.subtract(centroids[i],temp_centroids[i]))**2
                a = temp3[i] * (norm(np.subtract(X[j],temp_centroids[i]))**2)
                p[i] = (l*p[i]) + (2*q*l) + (b*m[i]*l) + a
                g[i] = (l*g[i]) + np.multiply((l*m[i]),np.subtract(centroids[i],temp_centroids[i])) + np.multiply(temp3[i],np.subtract(X[j],temp_centroids[i]))
            centroids = temp_centroids.copy()
            eK[j-K] = np.sum(p)
        return eK

#Function-5: Finding d_max for exponentially fading setting
def dmax(K,X,l):
    if len(X)<K:
        print("length is an issue")
    else:
        d = []
        centroids = X[:K]
        counter = np.ones(K)
        for x_n in X[K:]:
            closest_k = find_closest_cluster(x_n,centroids)
            counter[closest_k] +=1
            temp = np.subtract(x_n,centroids[closest_k])
            temp = np.multiply(temp,l)
            centroids[closest_k] = np.add(centroids[closest_k],temp)
            for h in range(len(centroids)):
                temp2 = 0
                for x in range(len(centroids)):
                    v = np.linalg.norm(np.subtract(centroids[h],centroids[x]))
                    if v>temp2:
                        temp2 = v
            d = d + [temp2]
        return d
#Function-6: Finding E_K for general setting
def eK(K,X):
    if len(X)<K:
        print("length is an issue")
    else:
        centroids = X[:K]
        counter = np.ones(K)
        p = np.zeros(K)
        temp_len = len(X) - K
        eK = np.zeros(temp_len)
        y = len(X[K])
        temp_len = (K,y)
        g = np.zeros(temp_len)
        
        for j in range(K,len(X),1):      
            closest_K = find_closest_cluster(X[j],centroids)
            m = counter.copy()
            counter[closest_K] +=1
            
            temp3 = np.subtract(counter,m)
            temp_centroids = centroids.copy()
            temp = np.subtract(X[j],temp_centroids[closest_K])
            temp = np.divide(temp, counter[closest_K])
            temp_centroids[closest_K] = np.add(temp_centroids[closest_K],temp)
            #update stage
            for i in range(K):
                #print("Cluster " + str(i))
                temp1 = np.subtract(centroids[i],temp_centroids[i])
                
                temp1 = np.transpose(temp1)
                q = np.matmul(temp1,g[i])
                #print(q[i])
                b = norm(np.subtract(centroids[i],temp_centroids[i]))**2
                #print(b[i])
                a = temp3[i] * (norm(np.subtract(X[j],temp_centroids[i]))**2)
                #print(a[i])
                p[i] = p[i] + (2*q) + (b*m[i]) + a
                #print(p[i])
                g[i] = g[i] + np.multiply(m[i],np.subtract(centroids[i],temp_centroids[i])) + np.multiply(temp3[i],np.subtract(X[j],temp_centroids[i]))
                
            centroids = temp_centroids.copy()
        
        
            eK[j-K] = np.sum(p)
        return eK
#Function-7: Calculating d_max in general setting
def dmax(K,X):
    if len(X)<K:
        print("length is an issue")
    else:
        d = []
        centroids = X[:K]
        counter = np.ones(K)
        for x_n in X[K:]:
            closest_k = find_closest_cluster(x_n,centroids)
            counter[closest_k] +=1
            temp = np.subtract(x_n,centroids[closest_k])
            temp = np.divide(temp,counter[closest_k])
            centroids[closest_k] = np.add(centroids[closest_k],temp)
            for h in range(len(centroids)):
                temp2 = 0
                for x in range(len(centroids)):
                    v = np.linalg.norm(np.subtract(centroids[h],centroids[x]))
                    if v>temp2:
                        temp2 = v
            d = d + [temp2]
        return d
#Function-8: Generating plot for I2-I11 in general setting
def final(name,data):
    t = np.divide(eK(1,data)[1:],eK(2,data))
    d = dmax(2,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/4
    I2= t

    t = np.divide(eK(1,data)[2:],eK(3,data))
    d = dmax(3,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/9
    I3= t
    t = np.divide(eK(1,data)[3:],eK(4,data))
    d = dmax(4,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/16
    I4= t  
    t = np.divide(eK(1,data)[4:],eK(5,data))
    d = dmax(5,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/25
    I5= t
    t = np.divide(eK(1,data)[5:],eK(6,data))
    d = dmax(6,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/36
    I6 = t
    t = np.divide(eK(1,data)[6:],eK(7,data))
    d = dmax(7,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/49
    I7= t
    t = np.divide(eK(1,data)[7:],eK(8,data))
    d = dmax(8,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/64
    I8= t
    t = np.divide(eK(1,data)[8:],eK(9,data))
    d = dmax(9,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/81
    I9= t
    t = np.divide(eK(1,data)[9:],eK(10,data))
    d = dmax(10,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/100
    I10= t
    t = np.divide(eK(1,data)[10:],eK(11,data))
    d = dmax(11,data)
    t = np.multiply(t,np.multiply(d,d))
    t = t/121
    I11= t

    plt.plot(I2[9:],label = "cluster-2")
    plt.plot(I3[8:],label = "cluster-3")
    plt.plot(I4[7:],label = "cluster-4")
    plt.plot(I5[6:],label = "cluster-5")
    plt.plot(I6[5:],label = "cluster-6")
    plt.plot(I7[4:],label = "cluster-7")
    plt.plot(I8[3:],label = "cluster-8")
    plt.plot(I9[2:],label = "cluster-9")
    plt.plot(I10[1:],label = "cluster-10")
    plt.plot(I11,label = "cluster-11")


    plt.legend()
    plt.savefig(name)
    
#Function-9: I2-I8 Plot in exponential weight decay setting:
def final(l,name,data):
    t = np.divide(eK(1,data,l)[1:],eK(2,data,l))
    d = dmax(2,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/4
    I2= t

    t = np.divide(eK(1,data,l)[2:],eK(3,data,l))
    d = dmax(3,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/9
    I3= t
    t = np.divide(eK(1,data,l)[3:],eK(4,data,l))
    d = dmax(4,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/16
    I4= t  
    t = np.divide(eK(1,data,l)[4:],eK(5,data,l))
    d = dmax(5,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/25
    I5= t
    t = np.divide(eK(1,data,l)[5:],eK(6,data,l))
    d = dmax(6,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/36
    I6 = t
    t = np.divide(eK(1,data,l)[6:],eK(7,data,l))
    d = dmax(7,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/49
    I7= t
    t = np.divide(eK(1,data,l)[7:],eK(8,data,l))
    d = dmax(8,data,l)
    t = np.multiply(t,np.multiply(d,d))
    t = t/64
    I8= t
    plt.plot(I2[10:],label = "cluster-2")
    plt.plot(I3[9:],label = "cluster-3")
    plt.plot(I4[8:],label = "cluster-4")
    plt.plot(I5[7:],label = "cluster-5")
    plt.plot(I6[6:],label = "cluster-6")
    plt.plot(I7[5:],label = "cluster-7")
    plt.plot(I8[4:],label = "cluster-8")
    plt.legend()
    plt.savefig(name)  