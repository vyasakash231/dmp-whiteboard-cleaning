#! /usr/bin/python3
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../")))

from basic_import import *
from .orientation_dmp import OrientationDMP


# DMP Explained : https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics/
class QuaternionDMP(OrientationDMP):
    def __init__(self, no_of_basis_func, dt=0.01, T=1, q_0=None, alpha=3, K=1050, D=None, W=None):
        super().__init__(no_of_basis_func, dt=dt, T=T, q_0=q_0, alpha=alpha, K=K, D=D, W=W)

    ####################################################################################################################################################

    def learn_dynamics(self, time, q_des, omega_des, tau=1):
        """
        Takes in a desired trajectory and generates the set of system parameters that best realize this path.
        X_des: the desired trajectories of each DMP should be shaped [no_of_dmps, num_timesteps]
        """
        q_des = self.make_quat_continuity(q_des)   # Important to make quaternions continuous 

        # set initial and goal state
        self.q_0 = q_des[:,[0]].copy()  # [[x0],[y0]]
        self.q_goal = q_des[:,[-1]].copy()  # [[xn],[yn]]

        # Ensure they are normalized
        self.q_0 = self.q_0 / np.linalg.norm(self.q_0)
        self.q_goal = self.q_goal / np.linalg.norm(self.q_goal)

        # start time from 0
        time = (time - time[0]).reshape(-1)
        
        # Normalize the time to [0, 1] and create uniform time steps
        max_time = time[-1]
        normalized_time = time / max_time
        uniform_time = np.linspace(0, 1, self.cs.time_steps)

        # -------------------- plot uniform trajectory from demo data --------------------
        path = np.zeros((4, self.cs.time_steps))
        for i in range(4):
            path_gen = scipy.interpolate.interp1d(normalized_time, q_des[i,:], kind="quadratic")
            path[i,:] = path_gen(uniform_time)

        # Evaluation of the interpolant
        q_des = path  # (4, N)

        path = np.zeros((3, self.cs.time_steps))
        for i in range(3):
            path_gen = scipy.interpolate.interp1d(normalized_time, omega_des[i,:], kind="quadratic")
            path[i,:] = path_gen(uniform_time)

        # Evaluation of the interpolant
        omega_des = path   # η = omega -> (3, N)
        omega_des[:,-1] = np.zeros(3)   # make final velocity 0

        # calculate acceleration of y_des (gradient of dX_des is computed using second order accurate central differences)
        omega_dot_des = np.empty_like(omega_des)
        for i in range(3):
            omega_dot_des[i,:] = np.gradient(omega_des[i,:]) / self.cs.dt  # (3, N)
        omega_dot_des[:,-1] = np.zeros(3)  # make final acceleration 0   

        theta_track = self.cs.rollout(tau=tau)
    
        ## Find the force required to move along this trajectory
        f_target = np.zeros([3, self.cs.time_steps])
        log_q_des = np.zeros((3, q_des.shape[1]))   # (3, N)
        
        for i in range(f_target.shape[1]):
            q_conj = self.quaternion_conjugate(q_des[:,i])  # shape (4,)
            q_product = self.quaternion_multiply(self.q_goal, q_conj)  # shape (4,)
            log_q_des[:,i] = self.quaternion_logarithm(q_product)  # shape (3,)
        
        # Calculate forcing term as per paper equation (16)
        for idx in range(3):
            f_target[idx,:] = tau * tau * omega_dot_des[idx,:] - 2 * self.K * log_q_des[idx,:] + tau * (self.D * omega_des[idx,:])  # (101,2)
        
        # generate weights to realize f_target
        self.generate_weights(f_target, theta_track)
        self.reset_state()

    def rollout(self, q_goal):
        """Generate a system trial, no feedback is incorporated."""
        self.reset_state()

        # set up tracking vectors
        y_track = np.zeros((4, self.cs.time_steps))
        dy_track = np.zeros((3, self.cs.time_steps))
      
        for t in range(self.cs.time_steps):
            y_track[:,[t]], dy_track[:,[t]] = self.step(q_goal)   # run and record timestep
        return y_track, dy_track
    
    def step(self, q_0=None, q_goal=None, epsilon=None, tau=1):
        """
        ----------------------- Run the DMP system for a single timestep ---------------------------
        Based on eqn (43) from the paper, Adaptation of manipulation skills in physical contact
        with the environment to reference force profiles
        τ^2*dω = 2*K*log(q_goal x q_conj) - D*(τ*ω) + f(θ)   ----- eqn (8)
        τ*dq = 0.5*(S(τ*ω) x q)                              ----- eqn (9)

        where, ω = 2*log(q_goal x q_conj

        In matrix form; q_product = q_goal x q_conj
            |dω_x|       |K 0 0|   log(q_product)_x   |D 0 0|   |ω_x|   |f_x|
        τ * |dω_y| = 2 * |0 K 0| * log(q_product)_y - |0 D 0| * |ω_y| + |f_y|       
            |dω_z|       |0 0 K|   log(q_product)_z   |0 0 D|   |ω_z|   |f_z|      

            |dq_x|         |0  -ωx -ωy -ωz|   |q_x|
        τ * |dq_y| = 0.5 * |ωx  0   ωz -ωy| * |q_y|
            |dq_z|         |ωy -ωz  0   ωx|   |q_z|
            |dq_w|         |ωz  ωy -ωx   0|   |q_w|
        """
        if q_0 is None:
            q_0 = self.q_0   # if initial state is not changed then keep is same as previous one

        K_matrix = self.K * np.eye(3)
        D_matrix = self.D * np.eye(3)

        psi = self.gaussian_basis_func(self.cs.theta)  # update basis function
        
        # Ensure current & goal quaternion is normalized
        self.q = self.q / np.linalg.norm(self.q)
        q_goal = q_goal / np.linalg.norm(q_goal)
        
        # update forcing term using weights learnt while imitating given trajectory
        sum_psi = np.sum(psi[:,[0]])
        if np.abs(sum_psi) <= 1e-6:  # avoid division by 0
            f = 0.0 * np.dot(self.W, psi[:,[0]])
        else:
            q0_conj = self.quaternion_conjugate(q_0)
            q0_product = self.quaternion_multiply(q_goal, q0_conj)
            log_q0 = self.quaternion_logarithm(q0_product)
            D = np.diag(log_q0)  # shape (3,3)
            f = D @ (np.dot(self.W, psi[:,[0]]) / sum_psi) * self.cs.theta

        # Solve System of Equations
        q_conj = self.quaternion_conjugate(self.q)
        q_product = self.quaternion_multiply(q_goal, q_conj)
        log_q = self.quaternion_logarithm(q_product)
    
        self.omega_dot = (1/tau*tau) * (2 * K_matrix @ log_q[:, np.newaxis] - tau * (D_matrix @ self.omega) + f)   

        # q(t+1) = χ(Δt*η/τ) * q(t) = exp(Δt*(η/τ)) * q(t)
        r = self.quaternion_exp(0.5 * self.cs.dt * self.omega)
        q_new = self.quaternion_multiply(r, self.q)   # from eqn (23)
        
        # make quaternion continuous
        if np.dot(self.q.reshape(-1), q_new) < 0:  # Angle > 90 degrees
            self.q = -q_new[:,np.newaxis]  # Flip to maintain continuity
        else:
            self.q = q_new[:,np.newaxis]

        self.omega = self.omega + self.omega_dot * self.cs.dt
        
        # dist = self.distance_matrice(q_robot, self.q)
        self.cs.step(epsilon, tau=tau)  # update theta
        return self.q, self.omega
    
    
