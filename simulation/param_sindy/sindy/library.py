import numpy as np

FEATURE_NAMES = [
    '1','u_v','u_w','v','w','v^2','w^2','v*w','u_v^2','u_w^2','u_v*u_w',
    'v*u_v','v*u_w','w*u_v','w*u_w','v*u_v^2','w*u_w^2','v*u_v*u_w','w*u_v*u_w',
    'v^2*u_v','w^2*u_w','sin(theta)','cos(theta)','sin(theta)*u_v','cos(theta)*u_w'
]

def build_feature_matrix(X,U):
    X=np.asarray(X,float); U=np.asarray(U,float)
    th=X[:,2]; v=X[:,3]; w=X[:,4]; uv=U[:,0]; uw=U[:,1]
    cols=[np.ones(len(X)),uv,uw,v,w,v**2,w**2,v*w,uv**2,uw**2,uv*uw,
          v*uv,v*uw,w*uv,w*uw,v*uv**2,w*uw**2,v*uv*uw,w*uv*uw,
          v**2*uv,w**2*uw,np.sin(th),np.cos(th),np.sin(th)*uv,np.cos(th)*uw]
    return np.column_stack(cols), FEATURE_NAMES
