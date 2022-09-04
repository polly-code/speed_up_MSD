import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
plt.plot(np.arange(0,100), np.arange(0,100)*10)
plt.xlabel("Time, s")
plt.ylabel("Distance, m")
plt.savefig("straight_motion.png", dpi=300)
plt.show()

def straight_msd(x, y):
    msd = np.zeros((len(x), 2))
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            msd[j - i, 0] += (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            msd[j - i, 1] += 1
    return np.divide(
        msd[:, 0],
        msd[:, 1],
        out=np.zeros_like(msd[:, 0]),
        where=msd[:, 1] != 0,
    )

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
axs[0].plot(np.arange(1, 100), straight_msd(np.arange(100) * 10, np.zeros(100))[1:])
axs[0].set_xlabel("Time, s")
axs[0].set_ylabel(r"MSD, $m^2$")

axs[1].plot(
    np.arange(1, 100),
    straight_msd(np.arange(100) * 10, np.zeros(100))[1:],
    label="simulation",
)
axs[1].plot(
    np.arange(1, 100), (np.arange(1, 100) * 10) ** 2, "--", label=r"theory $\alpha=2$"
)
axs[1].set_xlabel("Time, s")
axs[1].set_ylabel(r"MSD, $m^2$")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].legend()
plt.savefig("straight_motion_msd+log.png", dpi=300)
plt.show()

def random_walk_2d(n=100):
    r=10
    phi = np.random.uniform(0, 2*np.pi, n-1)
    x = np.zeros(n)
    y = np.zeros(n)
    
    x[1:] = r*np.cos(phi)
    y[1:] = r*np.sin(phi)
    return np.cumsum(x), np.cumsum(y)

def straight_msd(x, y):
    msd = np.zeros((len(x), 2))
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            msd[j - i, 0] += (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
            msd[j - i, 1] += 1
    return np.divide(
        msd[:, 0], 
        msd[:, 1], 
        out=np.zeros_like(msd[:, 0]), 
        where=msd[:, 1] != 0,
    )
            
            

x,y = random_walk_2d(n=1000)
fig, axs = plt.subplots(1,2, figsize=(14,5))
axs[0].plot(x,y)
axs[0].set_xlabel("X")
axs[0].set_ylabel(r"Y")

axs[1].plot(np.arange(len(x)), straight_msd(x,y), label='Random walk')
axs[1].plot(np.arange(1,len(x)), 100*np.arange(1,len(x)), label='a = 1')
axs[1].set_xlabel("Time, s")
axs[1].set_ylabel(r"MSD, $m^2$")
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].legend()
plt.savefig("traj_random_walk_msd_log.png", dpi=300)
plt.show()


def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate() # power spectrum density
    res = np.fft.ifft(PSD)
    res = (res[:N]).real #now we have the autocorrelation in convention B
    n = N * np.ones(N) - np.arange(0,N) #divide res(m) by (N-m)
    return res / n #this is the autocorrelation in convention A

def msd_fft(r):
    N = len(r)
    D = np.square(r).sum(axis=1) 
    D = np.append(D, 0) 
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2

r=np.array([x,y]).T
fast_msd = msd_fft(r)
slow_msd = straight_msd(x,y)
plt.figure(figsize=(7,5))
plt.plot(np.arange(1,len(x)), slow_msd[1:], label='Slow')
plt.plot(np.arange(1,len(x)), fast_msd[1:], label='Fast')

plt.xlabel("Time, s")
plt.ylabel(r"MSD, $m^2$")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig("comparison_methods.png", dpi=300)
plt.show()

# IPython magic block
# uncomment if you know what you are doing
#x,y = random_walk_2d(n=3000)
#r=np.array([x,y]).T
#%timeit msd_fft(r)
#%timeit straight_msd(x,y)
