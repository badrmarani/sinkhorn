import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def cost(n, m):
    return np.array([
        abs(i-j)
        for i, j in product(range(n), range(m))
    ]).reshape((n,m))

def cosine(u, v, eps=1e-8):
    uu = np.einsum("ij,ij->", u, u, optimize=True)
    vv = np.einsum("ij,ij->", v, v, optimize=True)
    uv = np.einsum("ij,ij->", u, v, optimize=True)

    deno = np.sqrt(uu) * np.sqrt(vv)
    return 1 - (uv / deno)

def gauss(m, s, N):
    x = np.arange(0, N, dtype=np.float64)
    h = np.exp(-(x-m)**2 / (2*s**2))
    return x, h/h.sum()

def sinkhorn(a, b, niter=100, reg=1, tol=1e-9, mode="naive"):
    n, m = len(a), len(b)
    a = a/a.max().astype(np.float64)
    b = b/b.max().astype(np.float64)
    if len(a.shape) <= 1:
        a = a.reshape((-1,1))
        b = b.reshape((-1,1))
    
    u = np.full((n,1), 1/n)
    v = np.full((m,1), 1/m)
    
    C = cost(n,m)

    lmd = np.exp(-1/reg)
    K = lmd ** C
    
    p, q = np.zeros(shape=(m,1)), np.zeros(shape=(m,1))
    r, s = np.zeros(shape=(n,1)), np.zeros(shape=(n,1))

    for t in range(niter):        
        if mode == "naive":
            u = a/np.dot(K, v)
            v = b/np.dot(K.T, u)
        elif mode == "horner":        
            p[0], q[n-1] = v[0], 0.
            for i in range(1, n):
                p[i] = lmd * p[i-1] + v[i]
                q[n-i-1] = lmd * (q[n-i] + v[n-i])

            pq = p+q
            u = a/pq
            
            r[0], s[m-1] = u[0], 0.
            for i in range(1, m):
                r[i] = lmd * r[i-1] + u[i]
                s[m-i-1] = lmd * (s[m-i] + u[m-i])
            
            rs = r+s
            v = b/rs
        else:
            raise NotImplemented

    P = u.reshape((-1,1))* K * v.reshape((1,-1))
    return P

#### Test

N, h = 250, 1
x = np.arange(0, N, h)
x, u = gauss(70, 50, N)
x, v = gauss(60, 20, N)

P = sinkhorn(u, v, niter=100, reg=20, mode="horner")


plt.imshow(P)
plt.show()