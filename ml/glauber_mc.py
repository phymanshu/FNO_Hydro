import torch, math
R_Pb=6.62; a_Pb=0.546; sigma_NN=6.4; A=208

def sample_woods_saxon(n_nucleons, n_events, R=R_Pb, a=a_Pb):
    all_pos=[]
    for _ in range(n_events):
        pos=[]
        while len(pos)<n_nucleons:
            r=torch.rand(n_nucleons*4)*2.5*R
            prob=1.0/(1.0+torch.exp((r-R)/a))
            mask=torch.rand(len(r))<prob
            r_acc=r[mask][:n_nucleons-len(pos)]
            if len(r_acc)>0:
                theta=torch.acos(1-2*torch.rand(len(r_acc)))
                phi=torch.rand(len(r_acc))*2*math.pi
                x=r_acc*torch.sin(theta)*torch.cos(phi)
                y=r_acc*torch.sin(theta)*torch.sin(phi)
                z=r_acc*torch.cos(theta)
                pos.extend(zip(x.tolist(),y.tolist(),z.tolist()))
        all_pos.append(torch.tensor(pos[:n_nucleons]))
    return torch.stack(all_pos)

def nucleons_to_field(pos_A,pos_B,b,nx=64,ny=64,xy_max=12.0,sigma=0.4):
    x=torch.linspace(-xy_max,xy_max,nx)
    y=torch.linspace(-xy_max,xy_max,ny)
    X,Y=torch.meshgrid(x,y,indexing="ij")
    xA=pos_A[:,0]+b/2; yA=pos_A[:,1]
    xB=pos_B[:,0]-b/2; yB=pos_B[:,1]
    d_max=math.sqrt(sigma_NN/math.pi)
    part_A=torch.zeros(A,dtype=torch.bool)
    part_B=torch.zeros(A,dtype=torch.bool)
    for i in range(A):
        dist=torch.sqrt((xA[i]-xB)**2+(yA[i]-yB)**2)
        if (dist<d_max).any():
            part_A[i]=True
            part_B[dist<d_max]=True
    eps=torch.zeros(nx,ny)
    norm=1.0/(2*math.pi*sigma**2)
    for xi,yi in zip(torch.cat([xA[part_A],xB[part_B]]).tolist(),
                     torch.cat([yA[part_A],yB[part_B]]).tolist()):
        eps+=norm*torch.exp(-((X-xi)**2+(Y-yi)**2)/(2*sigma**2))
    eps=eps/(eps.sum()+1e-10)
    x_cm=(X*eps).sum(); y_cm=(Y*eps).sum()
    Xc=X-x_cm; Yc=Y-y_cm
    r2=Xc**2+Yc**2; phi=torch.atan2(Yc,Xc)
    denom=(eps*r2).sum()+1e-10
    eps2=(torch.sqrt((eps*r2*torch.cos(2*phi)).sum()**2+
                     (eps*r2*torch.sin(2*phi)).sum()**2)/denom).item()
    r3=r2*torch.sqrt(r2); denom3=(eps*r3).sum()+1e-10
    eps3=(torch.sqrt((eps*r3*torch.cos(3*phi)).sum()**2+
                     (eps*r3*torch.sin(3*phi)).sum()**2)/denom3).item()
    npart=part_A.sum().item()+part_B.sum().item()
    return eps,eps2,eps3,npart
