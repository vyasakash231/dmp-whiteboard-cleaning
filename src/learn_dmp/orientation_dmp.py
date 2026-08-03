#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from common_utils import svd_solve
from .canonical_system import CanonicalSystem, CanonicalSystem_phase_stopping


# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class OrientationDMP:
    def __init__(self, no_of_basis_func, dt=0.01, T=1, q_0=None, alpha=3, K=1050, D=None, W=None):
        """
        no_of_DMPs         : number of dynamic movement primitives (i.e. dimensions)
        no_of_basis_func   : number of basis functions per DMP (actually, they will be one more)
        dt             : timestep for simulation
        q_0            : initial quaternion state of DMPs
        q_goal         : goal quaternion state of DMPs
        T              : final time
        K              : elastic parameter in the dynamical system
        D              : damping parameter in the dynamical system
        w              : associated weights
        alpha          : constant of the Canonical System
        """
        self.no_of_basis_func = no_of_basis_func

        # Set up the DMP system
        if q_0 is None:
            q_0 = np.zeros(4)
        self.q_0 = copy.deepcopy(q_0)

        self.K = K   # stiffness
        if D is None:
            self.D = 2.0 * np.sqrt(self.K)  # damping 
        else:
            self.D = D

        # self.cs = CanonicalSystem(dt=dt, alpha=alpha, run_time=T)  # setup a canonical system
        self.cs = CanonicalSystem_phase_stopping(dt=dt, alpha=alpha, beta=10.0, run_time=T)  # setup a canonical system
        
        self.reset_state()  # set up the DMP system

        self.center_of_gaussian()  # centers of Gaussian basis functions distributed along the phase of the movement
        self.variance_of_gaussian()  # width/variance of Gaussian basis functions distributed along the phase of the movement
        
        # If no weights are give, set them to zero (default, f = 0)
        if W is None:  
            W = np.zeros((3, self.no_of_basis_func))
        self.W = W

    def center_of_gaussian(self):
        self.c = np.exp(-self.cs.alpha * np.linspace(0, self.cs.run_time, self.no_of_basis_func + 1))  #  centers are exponentially spaced

    def variance_of_gaussian(self):
        """width/variance of gaussian distribution"""
        self.width = np.zeros(self.no_of_basis_func)
        for i in range(self.no_of_basis_func):
            self.width[i] = 1 / ((self.c[i+1] - self.c[i])**2)
        self.width = np.append(self.width, self.width[-1])                                                                                              

    def reset_state(self):
        """Reset the system state"""
        self.q = self.q_0.copy()
        self.omega = np.zeros((3, 1))
        self.omeaga_dot = np.zeros((3, 1))
        self.cs.reset()

    def gaussian_basis_func(self, theta):
        """Generates the activity of the basis functions for a given canonical system rollout"""
        c = np.reshape(self.c, [self.no_of_basis_func + 1, 1])
        h = np.reshape(self.width, [self.no_of_basis_func + 1, 1])
        Psi_basis_func = np.exp(-h * (theta - c)**2)
        return Psi_basis_func

    def make_quat_continuity(self, quats):
        R, C = quats.shape
        for i in range(1, np.maximum(R, C)):
            if np.dot(quats[:,i-1], quats[:,i]) < 0:  # Angle > 90 degrees
                quats[:,i] = -quats[:,i]  # Flip to maintain continuity
        return quats
    
    """from, eqn (10) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def quaternion_multiply(self, q1, q2):
        """
        Multiply two quaternions
        q = [x, y, z, w] = (u + v)
        where, u = [x, y, z], v = [w]
        """
        if q1.ndim == 2:
            q1 = q1.reshape(-1)
        if q2.ndim == 2:
            q2 = q2.reshape(-1)
        
        u1, v1 = q1[:3], q1[-1]
        u2, v2 = q2[:3], q2[-1]
        
        # quaternion product => q1 * q2 = (v1 + u1) * (v2 + u2)
        v = v1*v2 - np.dot(u1, u2)
        u = v1*u2 + v2*u1 + np.cross(u1, u2)
        q = np.append(u,v)

        # Normalize to avoid numerical drift
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        return q
    
    """from, eqn (11) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def quaternion_logarithm(self, q):
        """ 
        log(q) : S^3 → R^3 
        q = [x, y, z, w] = (u + v)
        where, u = [x, y, z], v = [w]
        """
        if q.ndim == 2:
            q = q.reshape(-1)

        u, v = q[:3], q[-1]
        u_norm = np.linalg.norm(u)
        
        if u_norm < 1e-10:  # Almost zero
            return np.array([0, 0, 0])
        
        u_unit = u / u_norm
        # angle = np.arccos(np.clip(v, -1, 1))
        angle = 2 * np.arctan2(u_norm, v)
        return angle * u_unit
    
    """from, eqn (13) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def quaternion_exp(self, r):
        r_norm = np.linalg.norm(r)
        
        if r_norm < 1e-10:  # Almost zero
            return np.array([0, 0, 0, 1])

        r_unit = r / r_norm
        u = np.sin(r_norm) * (r_unit)
        v = np.cos(r_norm)
        q = np.append(u, v)
        return q
    
    """conjugate of quaternion q = [x, y, z, w] is q* = [-x, -y, -z, w]"""
    def quaternion_conjugate(self, q):
        u, v = q[:3], q[-1]
        return np.append(-u, v)  
    
    def distance_matrice(self, q1, q2):
        q2_conj = self.quaternion_conjugate(q2)
        quat_mul = self.quaternion_multiply(q1, q2_conj)
        u, v = quat_mul[:3], quat_mul[-1]
        if v != -1 and np.all(u) != 0:
            quat_log = self.quaternion_logarithm(quat_mul)
            return np.linalg.norm(quat_log)
        else:
            return np.pi
        
    def skew_matrix(self, z):
        mat = np.zeros((3,3))

        if z.ndim == 2:
            z = z.reshape(-1)

        mat[0,1] = -z[2]  # ωz
        mat[0,2] = z[1]  # -ωy
        mat[1,0] = z[2]  # -ωz
        mat[1,2] = -z[0]  # ωx
        mat[2,0] = -z[1]  # ωy
        mat[2,1] = z[0]  # -ωx
        return mat

    """eqn (11) from the paper -> Closed-Loop Manipulator Control Using Quaternion Feedback""" 
    def skew4x4(self, z):
        mat = np.zeros((4,4))

        if z.ndim == 2:
            z = z.reshape(-1)

        mat[0,1:] = -z  # [-ωx,-ωy,-ωz]
        mat[1:,0] = z  # [ωx,ωy,ωz]
        mat[1:,1:] = self.skew_matrix(z)
        return mat
    
    def q_dot(self, omega, q):
        Skew_omega = self.skew4x4(omega)
        return 0.5 * (Skew_omega @ q)
    
    def mat_inv(self, D):
        if abs(np.linalg.det(D)) >= 1e-4:  # This was calculated based on LU-Decomposition which is numerically not very stable
            return svd_solve(D)  # SVD is more numerically stable when dealing with matrices that might be ill-conditioned
        else:
            return np.linalg.pinv(D, rcond=1e-5)   

    """from, eqn (15) of Adaptation of manipulation skills in physical contact with the environment to reference force profiles"""
    def generate_weights(self, f_target, theta_track):
        """
        Generate a set of weights over the basis functions such that the target forcing 
        term trajectory is matched (f_target - f(θ), shape -> [3 x time_steps])
                    / ∑ W * ψ(θ) \              / W.T @ ψ(θ) \       
        f(θ) = D @ |--------------| * θ => D @ |--------------| * θ  
                    \   ∑ ψ(θ)   /              \   ∑ ψ(θ)   /       
        
             /  ψ(θ)  \     
        W @ |----------| * θ = D^(-1) @ f(θ) = A
             \ ∑ ψ(θ) /            
                                 
                | /  ψ(θ)  \     |^(-1)
        W = A @ ||----------| * θ|
                | \ ∑ ψ(θ) /     |
        """
        # generate Basis functions
        psi = self.gaussian_basis_func(theta_track)  # (B,N)

        q_conj = self.quaternion_conjugate(self.q_0)  # shape (4,)
        q_product = self.quaternion_multiply(self.q_goal, q_conj)  # shape (4,)
        log_q0 = self.quaternion_logarithm(q_product)  # shape (3,)
     
        # scaling factor
        A = np.zeros_like(f_target)   # (3,N)
        for i in range(f_target.shape[1]):
            D = np.diag(log_q0)  # shape (3,3)
            A[:,[i]] = self.mat_inv(D) @ f_target[:,[i]]

        # calculate basis function weights using "linear regression"
        sum_psi = np.sum(psi,0)
        self.W = np.nan_to_num(A @ np.linalg.pinv((psi / sum_psi) * theta_track))   # (3,B) = (3, N) x (N, B)
