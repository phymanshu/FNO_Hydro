#!/usr/bin/env python3
"""Generate MC Glauber IC in SONG format for vHLLE icModel=2."""
import numpy as np, sys

A=208; R_A=6.62; d=0.546; sigma=6.4
NX=NY=61
x=np.linspace(-12,12,NX); y=np.linspace(-12,12,NY)

def woods_saxon(r): return 1.0/(1.0+np.exp((r-R_A)/d))

def sample_nucleons(A, seed):
    np.random.seed(seed)
    pos=[]
    while len(pos)<A:
        r_try=np.random.uniform(0,R_A+10*d,A*4)
        prob=r_try**2*woods_saxon(r_try); prob/=prob.max()
        acc=np.random.rand(len(r_try))<prob
        r_acc=r_try[acc]
        cos_t=np.random.uniform(-1,1,len(r_acc))
        phi_r=np.random.uniform(0,2*np.pi,len(r_acc))
        sin_t=np.sqrt(1-cos_t**2)
        for i in range(len(r_acc)):
            pos.append((r_acc[i]*sin_t[i]*np.cos(phi_r[i]),
                        r_acc[i]*sin_t[i]*np.sin(phi_r[i])))
            if len(pos)>=A: break
    return np.array(pos[:A])

def thickness(nuc, xg, yg):
    T=np.zeros((len(xg),len(yg)))
    sg=np.sqrt(sigma/(2*np.pi))
    for (nx,ny) in nuc:
        dx=xg[:,None]-nx; dy=yg[None,:]-ny
        T+=np.exp(-(dx**2+dy**2)/(2*sg**2))/(2*np.pi*sg**2)
    return T

def generate(b, seed, outfile):
    nuc_A=sample_nucleons(A,seed);   nuc_A[:,0]-=b/2
    nuc_B=sample_nucleons(A,seed+1); nuc_B[:,0]+=b/2
    T_A=thickness(nuc_A,x,y); T_B=thickness(nuc_B,x,y)
    p_A=1-(1-sigma*T_B/A)**A
    p_B=1-(1-sigma*T_A/A)**A
    npart=T_A*p_A+T_B*p_B
    eps=npart**(4.0/3.0)
    eps_max=eps.max()
    if eps_max>0: eps=eps/eps_max*50.0
    with open(outfile,"w") as f:
        f.write(f"{NX} {NY}\n")
        for ix in range(NX):
            for iy in range(NY):
                f.write(f"{x[ix]:.6f}  {y[iy]:.6f}  {eps[ix,iy]:.6f}\n")

if __name__=="__main__":
    b=float(sys.argv[1]); seed=int(sys.argv[2]); out=sys.argv[3]
    generate(b, seed, out)
    print(f"MC Glauber IC written: b={b:.3f} fm seed={seed} → {out}")
