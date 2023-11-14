import numpy as np
import math

class KF():

    def __init__(self,N_state,N_meas,N_deriv,dt):
        self.N_state = N_state # Number of state variables
        self.N_meas = N_meas # Number of measured variables
        self.N_deriv = N_deriv # Order of derivatives
        self.dt = dt # delta time step
        self.construct_matrices()

    def construct_matrices(self):
        self.F = np.zeros((self.N_state*self.N_deriv,self.N_state*self.N_deriv))
        np.fill_diagonal(self.F,1.0)
        for n in range(1,self.N_deriv):
            np.fill_diagonal(self.F[(self.N_state*n):,:((self.N_deriv-n)*self.N_state)],np.power(self.dt,n)/(math.factorial(n)*1.0))
        self.H = np.zeros((self.N_meas,self.N_state*self.N_deriv))
        # Assuming that the state and not its derivatives are being measured
        np.fill_diagonal(self.H[:,(self.N_state*(self.N_deriv-1)):],1.0)
        self.Q = np.eye(self.N_state*self.N_deriv,self.N_state*self.N_deriv)
        self.R = np.eye(self.N_meas)

    def filter_data(self,Y_in,x_init,Q_vec):
        try:
            self.y = Y_in
            self.N_samples = Y_in.shape[1]
        except:
            self.N_samples = Y_in.shape[0]
            self.y = np.zeros((1,self.N_samples))
            self.y[0,:] = Y_in
        self.x_init = x_init
        self.Q_vec = Q_vec
        np.fill_diagonal(self.Q,Q_vec)
        # Forward filter step
        self.filter()
        # Return filtered data
        return self.x_s

    def forward_step(self,ind):
        self.x = np.dot(self.F,self.x)
        self.P = np.dot(self.F,np.dot(self.P,np.transpose(self.F)))+self.Q
        self.x_min[:,ind] = self.x
        self.P_min[:,:,ind] = self.P
        K = np.transpose(np.linalg.solve(np.transpose(np.dot(self.H,np.dot(self.P,np.transpose(self.H)))+self.R),np.transpose(np.dot(self.P,np.transpose(self.H)))))
        self.x = self.x+np.dot(K,self.y[:,ind]-np.dot(self.H,self.x))
        self.P = np.dot(np.eye(self.N_state*self.N_deriv)-np.dot(K,self.H),self.P)
        self.x_plus[:,ind] = self.x
        self.P_plus[:,:,ind] = self.P

    def backward_step(self,ind):
        if ind==(self.N_samples-1):
            self.x_s[:,ind] = self.x_plus[:,ind]
            self.P_s[:,:,ind] = self.P_plus[:,:,ind]
        else:
            A = np.transpose(np.linalg.solve(np.transpose(self.P_min[:,:,ind+1]),np.transpose(np.dot(self.P_plus[:,:,ind],np.transpose(self.F)))))
            self.x_s[:,ind] = self.x_plus[:,ind]+np.dot(A,self.x_s[:,ind+1]-self.x_min[:,ind+1])
            self.P_s[:,:,ind] = self.P_plus[:,:,ind]+np.dot(A,np.dot(self.P_s[:,:,ind+1]-self.P_min[:,:,ind+1],np.transpose(A)))

    def filter(self):
        self.x = self.x_init
        self.P = np.eye(self.N_state*self.N_deriv)
        self.x_min = np.zeros((self.N_state*self.N_deriv,self.N_samples))
        self.P_min = np.zeros((self.N_state*self.N_deriv,self.N_state*self.N_deriv,self.N_samples))
        self.x_plus = np.zeros((self.N_state*self.N_deriv,self.N_samples))
        self.P_plus = np.zeros((self.N_state*self.N_deriv,self.N_state*self.N_deriv,self.N_samples))
        # Iterate over samples and do the forward smoothing:
        for i in range(self.N_samples):
            self.forward_step(i)
        self.x_s = np.zeros((self.N_state*self.N_deriv,self.N_samples))
        self.P_s = np.zeros((self.N_state*self.N_deriv,self.N_state*self.N_deriv,self.N_samples))
        for j in range(self.N_samples-1,-1,-1):
            self.backward_step(j)
        # Finished

# -------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create Noisy Dataset:

    dt = 0.001
    t_vec = np.arange(0.0,10.0,dt)
    N_samples = t_vec.shape[0]
    print(N_samples)
    x_func = 2.0*np.cos(2*np.pi*t_vec)
    y_func = 2.0*np.sin(2*np.pi*t_vec)
    z_func = t_vec

    x_raw = x_func+0.05*np.random.randn(N_samples)
    y_raw = y_func+0.05*np.random.randn(N_samples)
    z_raw = z_func+0.05*np.random.randn(N_samples)

    # Filter data
    Y_raw = np.vstack((x_raw,y_raw,z_raw))

    K_filter = KF(3,3,3,dt)
    x_init = np.array([0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0])
    #q_vec = np.array([dt**2, dt**2, dt**2, 1.0, 1.0, 1.0, 1.0/dt**2, 1.0/dt**2, 1.0/dt**2])
    q_vec = 0.01*np.array([dt**2, dt**2, dt**2, dt, dt, dt, 1.0, 1.0, 1.0])
    X_smooth = K_filter.filter_data(Y_raw,x_init,q_vec)

    print(X_smooth.shape)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot 3D
    ax.plot3D(x_func,y_func,z_func, 'b')
    #ax.scatter(x_raw,y_raw,z_raw, '.')
    #ax.scatter(Y_raw[0,:],Y_raw[1,:],Y_raw[2,:],'.')
    ax.plot3D(X_smooth[6,:],X_smooth[7,:],X_smooth[8,:],'r')

    #ax.set_xlim([-3.0,3.0])
    #ax.set_ylim([-3.0,3.0])
    #ax.set_zlim([-3.0,3.0])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
