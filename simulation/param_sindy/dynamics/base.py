import numpy as np

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def base_dynamics(x,u,dt,friction_v=1.0,friction_w=1.0,slip_gain=0.0):
    x=np.asarray(x,float); u=np.asarray(u,float)
    if x.ndim==2:
        px,py,th,v,w=[x[:,i] for i in range(5)]
        uv,uw=u[:,0],u[:,1]
        v_next=np.clip(v + (friction_v*uv-v)*(dt/0.12), -2.0, 2.0)
        w_next=np.clip(w + (friction_w*uw-w)*(dt/0.10), -1.5, 1.5)
        slip=slip_gain*uv*uw
        out=x.copy()
        out[:,0]=px+(v_next*np.cos(th)-slip*np.sin(th))*dt
        out[:,1]=py+(v_next*np.sin(th)+slip*np.cos(th))*dt
        out[:,2]=wrap_angle(th+w_next*dt)
        out[:,3]=v_next; out[:,4]=w_next
        return out
    px,py,th,v,w=[float(z) for z in x]
    v_next=np.clip(v + (friction_v*u[0]-v)*(dt/0.12), -2.0, 2.0)
    w_next=np.clip(w + (friction_w*u[1]-w)*(dt/0.10), -1.5, 1.5)
    slip=slip_gain*u[0]*u[1]
    px += (v_next*np.cos(th) - slip*np.sin(th))*dt
    py += (v_next*np.sin(th) + slip*np.cos(th))*dt
    th = wrap_angle(th+w_next*dt)
    return np.array([px,py,th,v_next,w_next])

def nominal_dynamics(x,u,dt):
    return base_dynamics(x,u,dt,1.0,1.0,0.0)
