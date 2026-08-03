import numpy as np

class CanonicalSystem:
    """
    τ * (dθ/dt) = - alpha * θ
    
    Intergrate both sides;
    ∫(dθ/θ) = -(alpha/τ) * ∫dt
    ln(θ2) - ln(θ1) = -(alpha/τ) * (t2 - t1)

    ln(θ2/θ1) = -(alpha/τ) * (t2 - t1)
    θ2/θ1 = exp(-(alpha/τ) * dt)
    θ2 = exp(-(alpha/τ) * dt) * θ1
    """
    def __init__(self, dt, alpha, run_time=1):
        self.dt = dt
        self.alpha = alpha
        self.run_time = run_time  # T
        self.time_steps = int(self.run_time/self.dt)  # T/dt = 1/0.005 = 200 time steps

        self.reset()

    def reset(self):
        """Reset the system state"""
        self.theta = 1  # at t = 0, theta = 1

    def step(self,tau=1):
        """Perform single step integration"""
        self.theta = np.exp(-(self.alpha/tau)*self.dt) * self.theta
        # self.theta = self.theta - (self.alpha/tau) * self.theta * self.dt

    def rollout(self,tau=1):
        if tau != 0:
            timesteps = int(self.time_steps / tau)
        else:
            timesteps = self.time_steps

        theta_track = np.zeros(timesteps)
        self.reset()

        for i in range(timesteps):
            theta_track[i] = self.theta
            self.step(tau)

        return theta_track


class CanonicalSystem_phase_stopping:
    """
    Note that in the case of large force or torque errors, the error value ε becomes large which in turn makes 
    the phase change (dθ/dt) small. Thus the phase evolution is stopped until the robot reduces the force/torque error.

    τ * (dθ/dt) = - alpha * θ / (1 + beta * ε)
    where, ε = ||X - X_dmp|| + gamma * d(q x q_dmp_conj) 

    Intergrate both sides;
    θ(t+1) = θ(t) + (dθ/dt) * dt
    θ(t+1) = θ(t) - ((alpha/τ) / (1 + beta * ε)) * θ(t) * dt
    """
    def __init__(self, dt, alpha, beta, run_time=1):
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.run_time = run_time  # T
        self.time_steps = int(self.run_time/self.dt)  # T/dt = 1/0.005 = 200 time steps

        self.reset()

    def reset(self):
        """Reset the system state"""
        self.theta = 1  # at t = 0, theta = 1

    def step(self,epsilon, tau=1):
        """Perform single step integration"""
        self.theta = self.theta - ((self.alpha/tau) / (1 + self.beta * epsilon)) * self.theta * self.dt  # for imitation, we include error term (for stopping phase evolution)

    def rollout(self,tau=1):
        if tau != 0:
            timesteps = int(self.time_steps / tau)
        else:
            timesteps = self.time_steps

        theta_track = np.zeros(timesteps)
        self.reset()

        for i in range(timesteps):
            theta_track[i] = self.theta
            self.theta = self.theta - (self.alpha/tau) * self.theta * self.dt  # for learning, we don't include error term

        return theta_track
    