from dataclasses import dataclass
import numpy as np
from .library import build_feature_matrix
from ..dynamics.base import nominal_dynamics, wrap_angle

@dataclass
class RLSModel:
    Theta: np.ndarray
    P: np.ndarray
    alpha: float
    forgetting: float
    version: int = 1

    def predict_next(self, x, u, dt):
        x=np.asarray(x,float); u=np.asarray(u,float); phi,_=build_feature_matrix(x if x.ndim==2 else x[None,:],u if u.ndim==2 else u[None,:]); z=phi@self.Theta
        if x.ndim==2:
            th=x[:,2]; q=x.copy(); q[:,3]=z[:,0]; q[:,4]=z[:,1]
            q[:,0]=x[:,0]+(z[:,0]*np.cos(th)-z[:,2]*np.sin(th))*dt
            q[:,1]=x[:,1]+(z[:,0]*np.sin(th)+z[:,2]*np.cos(th))*dt
            q[:,2]=wrap_angle(th+z[:,1]*dt); return q
        z=z.reshape(-1); th=x[2]; q=x.copy(); q[3]=z[0]; q[4]=z[1]
        q[0]=x[0]+(z[0]*np.cos(th)-z[2]*np.sin(th))*dt
        q[1]=x[1]+(z[0]*np.sin(th)+z[2]*np.cos(th))*dt
        q[2]=wrap_angle(th+z[1]*dt); return q

def _targets(X,Xn,dt):
    th=X[:,2]
    dx=(Xn[:,0]-X[:,0])/dt; dy=(Xn[:,1]-X[:,1])/dt
    slip=-np.sin(th)*dx+np.cos(th)*dy
    return np.column_stack([Xn[:,3],Xn[:,4],slip])

def update_rls(model, x, u, xn, dt, forgetting=0.995, delta=10.0, alpha=0.20):
    phi,_=build_feature_matrix(np.asarray(x,float)[None,:],np.asarray(u,float)[None,:])
    y=_targets(np.asarray(x)[None,:],np.asarray(xn)[None,:],dt).reshape(-1)
    p=model.P; th=model.Theta
    ph=phi.reshape(-1,1)
    den=forgetting + (ph.T@p@ph).item()
    k=(p@ph)/den
    err=y-(phi@th).reshape(-1)
    th=th+k@err.reshape(1,-1)
    p=(p-k@(ph.T@p))/forgetting
    return RLSModel(th,p,float(alpha),forgetting,model.version+1)

def init_rls(n_features, n_targets=3, alpha=0.20, delta=10.0, forgetting=0.995):
    return RLSModel(np.zeros((n_features,n_targets)), np.eye(n_features)*delta, float(alpha), forgetting)
