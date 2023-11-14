import numpy as np
import os

# Extended Kalman Filter

class EKF():
    
    def __init__(self,dt):
        self.N_state = 13 # Number of state variables
        self.N_meas = 4 # Number of measured variables
        self.dt = dt # delta time step
        self.construct_matrices()

    def PHI_matrix(self,x_in):
        d_t = self.dt
        d_t2 = self.dt/2.0
        d_t4 = (self.dt**2)/4.0
        wx = x_in[6]
        wy = x_in[7]
        wz = x_in[8]
        q_norm = np.linalg.norm(x_in[9:13])
        q0 = x_in[9]/q_norm
        q1 = x_in[10]/q_norm
        q2 = x_in[11]/q_norm
        q3 = x_in[12]/q_norm
        #Phi = np.array([[1,0,0,0,0,0,0,0,0,0],
        #    [0,1,0,0,0,0,0,0,0,0],
        #    [0,0,1,0,0,0,0,0,0,0],
        #    [d_t,0,0,1,0,0,0,0,0,0],
        #    [0,d_t,0,0,1,0,0,0,0,0],
        #    [0,0,d_t,0,0,1,0,0,0,0],
        #    [0,0,0,-q1*d_t2,-q2*d_t2,-q3*d_t2,1      ,-wx*d_t2,-wy*d_t2,-wz*d_t2],
        #    [0,0,0, q0*d_t2,-q3*d_t2, q2*d_t2,wx*d_t2,1       ,-wz*d_t2, wy*d_t2],
        #    [0,0,0, q3*d_t2, q0*d_t2,-q1*d_t2,wy*d_t2, wz*d_t2,1       ,-wx*d_t2],
        #    [0,0,0,-q2*d_t2, q1*d_t2, q0*d_t2,wz*d_t2,-wy*d_t2,wx*d_t2,1       ]])
        #Phi = np.array([[1,0,0,0,0,0,0,0,0,0],
        #    [0,1,0,0,0,0,0,0,0,0],
        #    [0,0,1,0,0,0,0,0,0,0],
        #    [d_t,0,0,1,0,0,0,0,0,0],
        #    [0,d_t,0,0,1,0,0,0,0,0],
        #    [0,0,d_t,0,0,1,0,0,0,0],
        #    [0,0,0,-q1*d_t2,-q2*d_t2,-q3*d_t2,1      ,-wx*d_t2,-wy*d_t2,-wz*d_t2],
        #    [0,0,0, q0*d_t2,-q3*d_t2, q2*d_t2,wx*d_t2,1       , wz*d_t2,-wy*d_t2],
        #    [0,0,0, q3*d_t2, q0*d_t2,-q1*d_t2,wy*d_t2,-wz*d_t2,1       , wx*d_t2],
        #    [0,0,0,-q2*d_t2, q1*d_t2, q0*d_t2,wz*d_t2, wy*d_t2,-wx*d_t2,1       ]])
        #Phi = np.array([[1,0,0,0,0,0,0,0,0,0],
        #    [0,1,0,0,0,0,0,0,0,0],
        #    [0,0,1,0,0,0,0,0,0,0],
        #    [d_t,0,0,1,0,0,0,0,0,0],
        #    [0,d_t,0,0,1,0,0,0,0,0],
        #    [0,0,d_t,0,0,1,0,0,0,0],
        #    [0,0,0,-q1*d_t2,-q2*d_t2,-q3*d_t2,1      ,-wx*d_t2,-wy*d_t2,-wz*d_t2],
        #    [0,0,0, q0*d_t2, q3*d_t2,-q2*d_t2,wx*d_t2,1       ,-wz*d_t2, wy*d_t2],
        #    [0,0,0,-q3*d_t2, q0*d_t2, q1*d_t2,wy*d_t2, wz*d_t2,1       ,-wx*d_t2],
        #    [0,0,0, q2*d_t2,-q1*d_t2, q0*d_t2,wz*d_t2,-wy*d_t2, wx*d_t2,1       ]])
        Phi = np.array([
            [1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0,0],
            [d_t,0,0,1,0,0,0,0,0,0,0,0,0],
            [0,d_t,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,d_t,0,0,1,0,0,0,0,0,0,0],
            [(d_t**2)/2.0,0,0,d_t,0,0,1,0,0,0,0,0,0],
            [0,(d_t**2)/2.0,0,0,d_t,0,0,1,0,0,0,0,0],
            [0,0,(d_t**2)/2.0,0,0,d_t,0,0,1,0,0,0,0],
            [0,0,0,-q1*d_t4,-q2*d_t4,-q3*d_t4,-q1*d_t2,-q2*d_t2,-q3*d_t2,1      ,-wx*d_t2,-wy*d_t2,-wz*d_t2],
            [0,0,0, q0*d_t4, q3*d_t4,-q2*d_t4, q0*d_t2, q3*d_t2,-q2*d_t2,wx*d_t2,1       ,-wz*d_t2, wy*d_t2],
            [0,0,0,-q3*d_t4, q0*d_t4, q1*d_t4,-q3*d_t2, q0*d_t2, q1*d_t2,wy*d_t2, wz*d_t2,1       ,-wx*d_t2],
            [0,0,0, q2*d_t4,-q1*d_t4, q0*d_t4, q2*d_t2,-q1*d_t2, q0*d_t2,wz*d_t2,-wy*d_t2, wx*d_t2,1       ]])
        #Phi = np.array([
        #    [1,0,0,0,0,0,0,0,0,0,0,0,0],
        #    [0,1,0,0,0,0,0,0,0,0,0,0,0],
        #    [0,0,1,0,0,0,0,0,0,0,0,0,0],
        #    [d_t,0,0,1,0,0,0,0,0,0,0,0,0],
        #    [0,d_t,0,0,1,0,0,0,0,0,0,0,0],
        #    [0,0,d_t,0,0,1,0,0,0,0,0,0,0],
        #    [0,0,0,d_t,0,0,1,0,0,0,0,0,0],
        #    [0,0,0,0,d_t,0,0,1,0,0,0,0,0],
        #    [0,0,0,0,0,d_t,0,0,1,0,0,0,0],
        #    [0,0,0,0,0,0,-q1*d_t2,-q2*d_t2,-q3*d_t2,1      ,-wx*d_t2,-wy*d_t2,-wz*d_t2],
        #    [0,0,0,0,0,0, q0*d_t2, q3*d_t2,-q2*d_t2,wx*d_t2,1       ,-wz*d_t2, wy*d_t2],
        #    [0,0,0,0,0,0,-q3*d_t2, q0*d_t2, q1*d_t2,wy*d_t2, wz*d_t2,1       ,-wx*d_t2],
        #    [0,0,0,0,0,0, q2*d_t2,-q1*d_t2, q0*d_t2,wz*d_t2,-wy*d_t2, wx*d_t2,1       ]])
        return Phi

    def construct_matrices(self):
        self.H = np.array([[0,0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,1]])
        self.Q = np.eye(self.N_state)
        self.R = np.eye(self.N_meas)

    def filter_data(self,Y_in,x_init,Q_vec):
        self.y = Y_in
        self.N_samples = Y_in.shape[1]
        self.x_init = x_init
        phi_test = self.PHI_matrix(self.x_init)
        #print(self.x_init)
        #print(phi_test)
        #print(np.dot(phi_test,self.x_init))
        self.Q_vec = Q_vec
        np.fill_diagonal(self.Q,Q_vec)
        # Forward filter step
        self.filter()
        # Return filtered data
        return self.x_s

    def forward_step(self,ind):
        PHI = self.PHI_matrix(self.x)
        self.x = np.dot(PHI,self.x)
        x_norm = np.sqrt(np.sum(np.power(self.x[9:13],2)))
        if x_norm > 1.0e-3:
            self.x[9:13] = self.x[9:13]/x_norm
        self.P = np.dot(PHI,np.dot(self.P,np.transpose(PHI)))+self.Q
        self.x_min[:,ind] = self.x
        self.P_min[:,:,ind] = self.P
        K = np.transpose(np.linalg.solve(np.transpose(np.dot(self.H,np.dot(self.P,np.transpose(self.H)))+self.R),np.transpose(np.dot(self.P,np.transpose(self.H)))))
        self.x = self.x+np.dot(K,self.y[:,ind]-np.dot(self.H,self.x))
        x_norm = np.sqrt(np.sum(np.power(self.x[9:13],2)))
        if x_norm > 1.0e-3:
            self.x[9:13] = self.x[9:13]/x_norm
        self.P = np.dot(np.eye(self.N_state)-np.dot(K,self.H),self.P)
        self.x_plus[:,ind] = self.x
        self.P_plus[:,:,ind] = self.P

    def backward_step(self,ind):
        if ind==(self.N_samples-1):
            self.x_s[:,ind] = self.x_plus[:,ind]
            self.P_s[:,:,ind] = self.P_plus[:,:,ind]
        else:
            PHI = self.PHI_matrix(self.x_plus[:,ind])
            A = np.transpose(np.linalg.solve(np.transpose(self.P_min[:,:,ind+1]),np.transpose(np.dot(self.P_plus[:,:,ind],np.transpose(PHI)))))
            self.x_s[:,ind] = self.x_plus[:,ind]+np.dot(A,self.x_s[:,ind+1]-self.x_min[:,ind+1])
            x_norm = np.sqrt(np.sum(np.power(self.x_s[9:13,ind],2)))
            if x_norm > 1.0e-3:
                self.x_s[9:13,ind] = self.x_s[9:13,ind]/x_norm
            self.P_s[:,:,ind] = self.P_plus[:,:,ind]+np.dot(A,np.dot(self.P_s[:,:,ind+1]-self.P_min[:,:,ind+1],np.transpose(A)))

    def filter(self):
        self.x = self.x_init
        self.P = np.eye(self.N_state)
        self.x_min = np.zeros((self.N_state,self.N_samples))
        self.P_min = np.zeros((self.N_state,self.N_state,self.N_samples))
        self.x_plus = np.zeros((self.N_state,self.N_samples))
        self.P_plus = np.zeros((self.N_state,self.N_state,self.N_samples))
        # Iterate over samples and do the forward smoothing:
        for i in range(self.N_samples):
            self.forward_step(i)
        self.x_s = np.zeros((self.N_state,self.N_samples))
        self.P_s = np.zeros((self.N_state,self.N_state,self.N_samples))
        for j in range(self.N_samples-1,-1,-1):
            self.backward_step(j)
        # Finished
