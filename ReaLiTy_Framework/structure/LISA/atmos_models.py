import numpy as np
import torch
from scipy.special import gamma
from scipy.integrate import trapz
import PyMieScatt as ps

class LISA():
    def __init__(self,m=1.328,lam=905,rmax=200,rmin=1.5,bdiv=3e-3,dst=0.05,
                 dR=0.09,saved_model=False,atm_model='rain',mode='strongest'):
        self.m    = m
        self.lam  = lam
        self.rmax = rmax   # max range (m)
        self.bdiv = bdiv  # beam divergence (rad)
        self.dst  = dst   # min rain drop diameter to be sampled (mm)
        self.rmin = rmin   # min lidar range (bistatic)
        self.dR   = dR
        self.mode = mode
        self.atm_model = atm_model
        
        if saved_model:
            dat = np.load('mie_q.npz')
            self.D     = dat['D']
            self.qext  = dat['qext']
            self.qback = dat['qback']
        else:
            try:
                dat = np.load('mie_q.npz')
                self.D     = dat['D']
                self.qext  = dat['qext']
                self.qback = dat['qback']
            except:
                print('Calculating Mie coefficients... \nThis might take a few minutes')
                self.D,self.qext,self.qback = self.calc_Mie_params()
                print('Mie calculation done...')
        
        if atm_model=='rain':
            self.N_model = lambda D, Rr    : self.N_MP_rain(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MP_tot_rain(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MP_Sample_rain(Rr,N,dst)
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        
        elif atm_model=='snow':
            self.N_model = lambda D, Rr    : self.N_MG_snow(D,Rr)
            self.N_tot   = lambda Rr,dst   : self.N_MG_tot_snow(Rr,dst)
            self.N_sam   = lambda Rr,N,dst : self.MG_Sample_snow(Rr,N,dst)
            self.m       = 1.3031 # refractive index of ice
            self.augment  = lambda pc,Rr : self.augment_mc(pc,Rr)
        
        elif atm_model=='chu_hogg_fog':
            self.N_model = lambda D : self.Nd_chu_hogg(D)
            self.augment  = lambda pc : self.augment_avg(pc)
        
        elif atm_model=='strong_advection_fog':
            self.N_model = lambda D : self.Nd_strong_advection_fog(D)
            self.augment  = lambda pc : self.augment_avg(pc)
        
        elif atm_model=='moderate_advection_fog':
            self.N_model = lambda D : self.Nd_moderate_advection_fog(D)
            self.augment  = lambda pc : self.augment_avg(pc)

    # ==========================================
    # PYTORCH ACCELERATED MONTE CARLO ENGINE
    # ==========================================
    def augment_mc(self, pc_np, Rr):
        """Vectorized PyTorch implementation of the LISA Monte Carlo simulation"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. HUGE SPEEDUP: Calculate alpha ONCE per frame, not 120,000 times!
        Nd = self.N_model(self.D, Rr)
        alpha_np, _ = self.alpha_beta(Nd)
        alpha = float(alpha_np)

        pc = torch.tensor(pc_np, dtype=torch.float32, device=device)
        N_pts = pc.shape[0]

        # 2. Process in chunks to prevent VRAM Out-of-Memory on massive point clouds
        CHUNK_SIZE = 25000
        out_chunks = []

        for i in range(0, N_pts, CHUNK_SIZE):
            chunk = pc[i : i+CHUNK_SIZE]
            out_chunks.append(self._process_chunk(chunk, Rr, alpha, device))

        return torch.cat(out_chunks, dim=0).cpu().numpy()

    def _process_chunk(self, pc, Rr, alpha, device):
        x, y, z, ref = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
        
        # Load scalar configs
        rmax, bdiv, dst, n_refract, rmin, dR = float(self.rmax), float(self.bdiv), float(self.dst), float(self.m), float(self.rmin), float(self.dR)
        Pmin = 0.9 * (rmax**-2)

        # Vectorized Ranges and Powers
        ran = torch.sqrt(x**2 + y**2 + z**2)
        P0 = ref * torch.exp(-2 * alpha * ran) / (ran**2 + 1e-12)
        snr = P0 / Pmin

        Db = 1e3 * np.tan(bdiv) * ran
        bvol = (np.pi / 3) * ran * (1e-3 * Db / 2)**2

        # Calculate particles per ray
        N_tot_val = float(self.N_tot(Rr, dst))
        Nt_float = N_tot_val * bvol
        rand_frac = torch.rand_like(Nt_float)
        Nt = torch.floor(Nt_float) + (rand_frac < (Nt_float - torch.floor(Nt_float))).float()
        Nt = Nt.long()

        valid_mask = (ref != 0) & (ran > rmin)
        Nt[~valid_mask] = 0

        max_Nt = Nt.max().item()

        ran_new = torch.zeros_like(ran)
        ref_new = torch.zeros_like(ref)
        labl = torch.zeros_like(ref)

        if max_Nt > 0:
            N_pts = pc.shape[0]
            # Create a padded mask for vectorized particle sampling
            seq = torch.arange(max_Nt, device=device).unsqueeze(0).expand(N_pts, max_Nt)
            part_mask = seq < Nt.unsqueeze(1)

            U1 = torch.rand((N_pts, max_Nt), device=device)
            ran_r = ran.unsqueeze(1) * (U1 ** (1/3))
            part_mask = part_mask & (ran_r > rmin)

            U2 = torch.rand((N_pts, max_Nt), device=device)
            lmda = 4.1 * Rr**(-0.21) if self.atm_model == 'rain' else 2.55 * Rr**(-0.48)
            Dr = -torch.log(1 - U2 + 1e-12) / lmda + dst

            # Powers
            ref_r = abs((n_refract - 1) / (n_refract + 1))**2
            Db_ran_r = 1e3 * np.tan(bdiv) * ran_r
            Pr = ref_r * torch.exp(-2 * alpha * ran_r) * torch.clamp((Dr / (Db_ran_r + 1e-12))**2, max=1.0) / (ran_r**2 + 1e-12)
            Pr[~part_mask] = -1.0

            if self.mode == 'strongest':
                max_Pr, max_idx = torch.max(Pr, dim=1)
                max_ran_r = torch.gather(ran_r, 1, max_idx.unsqueeze(1)).squeeze(1)
                max_Dr = torch.gather(Dr, 1, max_idx.unsqueeze(1)).squeeze(1)
                max_Db = 1e3 * np.tan(bdiv) * max_ran_r

                cond_lost = (P0 < Pmin) & (max_Pr < Pmin)
                cond_scatter = (P0 < max_Pr) & (~cond_lost)
                cond_obj = (~cond_lost) & (~cond_scatter)

                # Assign Scatter
                ran_new[cond_scatter] = max_ran_r[cond_scatter]
                ref_new[cond_scatter] = ref_r * torch.exp(-2 * alpha * ran_new[cond_scatter]) * \
                                        torch.clamp((max_Dr[cond_scatter] / (max_Db[cond_scatter] + 1e-12))**2, max=1.0)
                labl[cond_scatter] = 1.0

                # Assign Object
                sig = dR / torch.sqrt(2 * snr[cond_obj] + 1e-12)
                ran_new[cond_obj] = ran[cond_obj] + torch.normal(mean=0.0, std=sig)
                ref_new[cond_obj] = ref[cond_obj] * torch.exp(-2 * alpha * ran[cond_obj])
                labl[cond_obj] = 2.0

            elif self.mode == 'last':
                cond_obj = P0 > Pmin
                cond_search = ~cond_obj

                sig = dR / torch.sqrt(2 * snr[cond_obj] + 1e-12)
                ran_new[cond_obj] = ran[cond_obj] + torch.normal(mean=0.0, std=sig)
                ref_new[cond_obj] = ref[cond_obj] * torch.exp(-2 * alpha * ran[cond_obj])
                labl[cond_obj] = 2.0

                valid_scatter = (Pr > Pmin) & part_mask
                ran_r_masked = torch.where(valid_scatter, ran_r, torch.tensor(-1.0, device=device))
                max_ran_r, max_idx = torch.max(ran_r_masked, dim=1)
                cond_scatter = cond_search & (max_ran_r > 0)

                if cond_scatter.any():
                    s_ran_r = torch.gather(ran_r, 1, max_idx.unsqueeze(1)).squeeze(1)
                    s_Dr = torch.gather(Dr, 1, max_idx.unsqueeze(1)).squeeze(1)
                    s_Db = 1e3 * np.tan(bdiv) * s_ran_r

                    ran_new[cond_scatter] = s_ran_r[cond_scatter]
                    ref_new[cond_scatter] = ref_r * torch.exp(-2 * alpha * ran_new[cond_scatter]) * \
                                            torch.clamp((s_Dr[cond_scatter] / (s_Db[cond_scatter] + 1e-12))**2, max=1.0)
                    labl[cond_scatter] = 1.0
        else:
            cond_obj = P0 >= Pmin
            sig = dR / torch.sqrt(2 * snr[cond_obj] + 1e-12)
            ran_new[cond_obj] = ran[cond_obj] + torch.normal(mean=0.0, std=sig)
            ref_new[cond_obj] = ref[cond_obj] * torch.exp(-2 * alpha * ran[cond_obj])
            labl[cond_obj] = 2.0

        # Angles and Reprojection
        valid_ran = ran > 0
        phi = torch.zeros_like(ran)
        the = torch.zeros_like(ran)

        phi[valid_ran] = torch.atan2(y[valid_ran], x[valid_ran])
        the[valid_ran] = torch.acos(torch.clamp(z[valid_ran] / ran[valid_ran], -1.0, 1.0))

        x_new = ran_new * torch.sin(the) * torch.cos(phi)
        y_new = ran_new * torch.sin(the) * torch.sin(phi)
        z_new = ran_new * torch.cos(the)

        return torch.stack((x_new, y_new, z_new, ref_new, labl), dim=1)

    # ==========================================
    # REMAINING ORIGINAL FUNCTIONS
    # ==========================================
    def augment_avg(self,pc):
        shp    = pc.shape      # data shape
        pc_new = np.zeros(shp) # init new point cloud
        leng   = shp[0]        # data length
        
        x    = pc[:,0]
        y    = pc[:,1]
        z    = pc[:,2]
        ref  = pc[:,3]          
        
        rmax = self.rmax       # max range (m)
        Pmin = 0.9*rmax**(-2)  # min measurable power (arb units)
        rmin = self.rmin       # min lidar range (bistatic)
        
        Nd          = self.N_model(self.D) # density of rain droplets (m^-3)
        alpha, beta = self.alpha_beta(Nd)  # extinction coeff. (1/m)  
        
        ran   = np.sqrt(x**2 + y**2 + z**2)  # range in m
        indx  = np.where(ran>rmin)[0]         # keep points where ranges larger than rmin
        
        P0        = np.zeros((leng,))                                  # init back reflected power
        P0[indx]  = ref[indx]*np.exp(-2*alpha*ran[indx])/(ran[indx]**2) # calculate reflected power
        snr       = P0/Pmin                                             # signal noise ratio
        
        indp = np.where(P0>Pmin)[0] # keep points where power is larger than Pmin
        
        sig        = np.zeros((leng,))                         # init sigma - std of range uncertainty
        sig[indp]  = self.dR/np.sqrt(2*snr[indp])                # calc. std of range uncertainty
        ran_new    = np.zeros((leng,))                         # init new range
        ran_new[indp]    = ran[indp] + np.random.normal(0,sig[indp])  # range with uncertainty added, keep range 0 if P<Pmin
        ref_new    = ref*np.exp(-2*alpha*ran)                   # new reflectance modified by scattering
        
        phi = np.zeros((leng,))
        the = np.zeros((leng,))
        
        phi[indx] = np.arctan2(y[indx],x[indx])   # angle in radians
        the[indx] = np.arccos(z[indx]/ran[indx])  # angle in radians
        
        pc_new[:,0] = ran_new*np.sin(the)*np.cos(phi)
        pc_new[:,1] = ran_new*np.sin(the)*np.sin(phi)
        pc_new[:,2] = ran_new*np.cos(the)
        pc_new[:,3] = ref_new
        
        return pc_new

    def msu_rain(self,pc,Rr):
        shp    = pc.shape      
        pc_new = np.zeros(shp) 
        leng   = shp[0]        
        
        x    = pc[:,0]
        y    = pc[:,1]
        z    = pc[:,2]
        ref  = pc[:,3]          
        
        rmax = self.rmax       
        Pmin = 0.9*rmax**(-2)/np.pi  
        
        alpha = 0.01* Rr**0.6
        
        ran      = np.sqrt(x**2 + y**2 + z**2)  
        indv     = np.where(ran>0)[0] 
        P0       = np.zeros((leng,))
        P0[indv] = ref[indv]*np.exp(-2*alpha*ran[indv])/(ran[indv]**2) 
        
        ran_new = np.zeros((leng,))
        ref_new = np.zeros((leng,))
        
        indp = np.where(P0>Pmin)[0] 
        ref_new[indp] = ref[indp]*np.exp(-2*alpha*ran[indp]) 
        sig = 0.02*ran[indp]* (1-np.exp(-Rr))**2
        ran_new[indp] = ran[indp] + np.random.normal(0,sig) 
        
        phi = np.zeros((leng,))
        the = np.zeros((leng,))
        
        phi[indp] = np.arctan2(y[indp],x[indp])   
        the[indp] = np.arccos(z[indp]/ran[indp])  
        
        pc_new[:,0] = ran_new*np.sin(the)*np.cos(phi)
        pc_new[:,1] = ran_new*np.sin(the)*np.sin(phi)
        pc_new[:,2] = ran_new*np.cos(the)
        pc_new[:,3] = ref_new
        
        return pc_new

    def haze_point_cloud(self,pts_3D,Rr=0):
        n = 0.05
        g = 0.35
        dmin = 2
            
        d = np.sqrt(pts_3D[:,0] * pts_3D[:,0] + pts_3D[:,1] * pts_3D[:,1] + pts_3D[:,2] * pts_3D[:,2])
        detectable_points = np.where(d>dmin)
        d = d[detectable_points]
        pts_3D = pts_3D[detectable_points]
        
        if (self.atm_model == 'rain') or (self.atm_model == 'snow'):
            Nd  = self.N_model(self.D,Rr) 
        elif (self.atm_model == 'chu_hogg_fog') or (self.atm_model=='strong_advection_fog') or (self.atm_model=='moderate_advection_fog'):
            Nd  = self.N_model(self.D) 
        else:
            print('Warning: weather model not implemented')
        alpha, beta = self.alpha_beta(Nd)     
    
        beta_usefull = alpha*np.ones(d.shape) 
        dmax = -np.divide(np.log(np.divide(n,(pts_3D[:,3] + g))),(2 * beta_usefull))
        dnew = -np.log(1 - 0.5) / (beta_usefull)
    
        probability_lost = 1 - np.exp(-beta_usefull*dmax)
        lost = np.random.uniform(0, 1, size=probability_lost.shape) < probability_lost
    
        cloud_scatter = np.logical_and(dnew < d, np.logical_not(lost))
        random_scatter = np.logical_and(np.logical_not(cloud_scatter), np.logical_not(lost))
        idx_stable = np.where(d<dmax)[0]
        old_points = np.zeros((len(idx_stable), 5))
        old_points[:,0:4] = pts_3D[idx_stable,:]
        old_points[:,3] = old_points[:,3]*np.exp(-beta_usefull[idx_stable]*d[idx_stable])
        old_points[:, 4] = np.zeros(np.shape(old_points[:,3]))
    
        cloud_scatter_idx = np.where(np.logical_and(dmax<d, cloud_scatter))[0]
        cloud_scatter = np.zeros((len(cloud_scatter_idx), 5))
        cloud_scatter[:,0:4] =  pts_3D[cloud_scatter_idx,:]
        cloud_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(cloud_scatter[:,0:3]), np.transpose(np.divide(dnew[cloud_scatter_idx],d[cloud_scatter_idx]))))
        cloud_scatter[:,3] = cloud_scatter[:,3]*np.exp(-beta_usefull[cloud_scatter_idx]*dnew[cloud_scatter_idx])
        cloud_scatter[:, 4] = np.ones(np.shape(cloud_scatter[:, 3]))
    
        random_scatter_idx = np.where(random_scatter)[0]
        scatter_max = np.min(np.vstack((dmax, d)).transpose(), axis=1)
        drand = np.random.uniform(high=scatter_max[random_scatter_idx])
        drand_idx = np.where(drand>dmin)
        drand = drand[drand_idx]
        random_scatter_idx = random_scatter_idx[drand_idx]
        fraction_random = .05 
        subsampled_idx = np.random.choice(len(random_scatter_idx), int(fraction_random*len(random_scatter_idx)), replace=False)
        drand = drand[subsampled_idx]
        random_scatter_idx = random_scatter_idx[subsampled_idx]
    
        random_scatter = np.zeros((len(random_scatter_idx), 5))
        random_scatter[:,0:4] = pts_3D[random_scatter_idx,:]
        random_scatter[:,0:3] = np.transpose(np.multiply(np.transpose(random_scatter[:,0:3]), np.transpose(drand/d[random_scatter_idx])))
        random_scatter[:,3] = random_scatter[:,3]*np.exp(-beta_usefull[random_scatter_idx]*drand)
        random_scatter[:, 4] = 2*np.ones(np.shape(random_scatter[:, 3]))
    
        dist_pts_3d = np.concatenate((old_points, cloud_scatter,random_scatter), axis=0)
    
        return dist_pts_3d
    
    def calc_Mie_params(self):
        out   = ps.MieQ_withDiameterRange(self.m, self.lam, diameterRange=(1,1e7),
                                        nd=2000, logD=True)
        D     = out[0]*1e-6
        qext  = out[1]
        qback = out[6]
        
        np.savez('mie_q.npz',D=D,qext=qext,qback=qback)
        
        return D,qext,qback
    
    def alpha_beta(self,Nd):
        D  = self.D
        qe = self.qext
        qb = self.qback
        alpha = 1e-6*trapz(D**2*qe*Nd,D)*np.pi/4 # m^-1
        beta  = 1e-6*trapz(D**2*qb*Nd,D)*np.pi/4 # m^-1
        return alpha, beta
    
    def N_MP_rain(self,D,Rr):
        return 8000*np.exp(-4.1*Rr**(-0.21)*D)
    
    def N_MP_tot_rain(self,Rr,dstart):
        lam = 4.1*Rr**(-0.21)
        return 8000*np.exp(-lam*dstart)/lam

    def MP_Sample_rain(self,Rr,N,dstart):
        lmda      = 4.1*Rr**(-0.21)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters
    
    def N_MG_snow(self,D,Rr):
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*D)
    
    def N_MG_tot_snow(self,Rr,dstart):
        N0   = 7.6e3* Rr**(-0.87)
        lmda = 2.55* Rr**(-0.48)
        return N0*np.exp(-lmda*dstart)/lmda

    def MG_Sample_snow(self,Rr,N,dstart):
        lmda      = 2.55* Rr**(-0.48)
        r         = np.random.rand(N)
        diameters = -np.log(1-r)/lmda + dstart
        return diameters

    def N_GD(self,D,rho,alpha,g,Rc):
        b = alpha/(g*Rc**g)
        Nd = g*rho*b**((alpha+1)/g)*(D/2)**alpha*np.exp(-b*(D/2)**g)/gamma((alpha+1)/g)
        return Nd

    def Nd_haze_coast(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=1,g=0.5,Rc=0.05e-3)
    
    def Nd_haze_continental(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=2,g=0.5,Rc=0.07)
    
    def Nd_strong_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=10)
    
    def Nd_moderate_advection_fog(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=3,g=1.,Rc=8)
    
    def Nd_strong_spray(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=4)
    
    def Nd_moderate_spray(self,D):
        return 1e9*self.N_GD(D*1e3,rho=100,alpha=6,g=1.,Rc=2)
    
    def Nd_chu_hogg(self,D):
        return 1e9*self.N_GD(D*1e3,rho=20,alpha=2,g=0.5,Rc=1)