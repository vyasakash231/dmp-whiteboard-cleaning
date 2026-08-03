import numpy as np

class Filters:
    def __init__(self):
        # Initialize torque filtering variables
        self.filtered_torque = None
        self.alpha = 0.25  # Filter coefficient (0.25 means 25% new, 75% previous)

        # Initialize moving average filter
        self.window_size = 5.0
        self.torque_history = []  # Changed from fixed-size list to empty list

    def low_pass_filter_torque(self, raw_torque):
        """ Butterworth-like low pass filter for signal smoothing """
        raw_torque = np.array(raw_torque)
        if self.filtered_torque is None:
            self.filtered_torque = raw_torque
            return raw_torque
            
        filtered_torque = self.alpha * raw_torque + (1 - self.alpha) * self.filtered_torque
        self.filtered_torque = filtered_torque
        return filtered_torque
    
    def moving_average_filter(self, raw_torque):
        """ Moving average filter with exponential weighting """
        # Add new torque to history
        self.torque_history.append(raw_torque)
        
        # Keep only window_size most recent values
        if len(self.torque_history) > self.window_size:
            self.torque_history.pop(0)
            
        # If history isn't full yet, return current torque
        if len(self.torque_history) < self.window_size:
            return raw_torque
            
        # Calculate exponentially weighted moving average
        weights = np.exp(np.linspace(-1, 0, len(self.torque_history)))
        weights = weights / np.sum(weights)
        
        filtered_torque = np.zeros_like(raw_torque)
        for i in range(len(self.torque_history)):
            filtered_torque += weights[i] * self.torque_history[i]
        return filtered_torque
    

class ButterWorthFilter:
    def __init__(self):
        # Butterworth 2nd Order Coefficients
        # Example: Cutoff 10Hz, Sampling 200Hz
        cutoff_freq = 10
        sampling_freq = 200 
        wc = np.tan(np.pi * cutoff_freq / sampling_freq)
        k1 = np.sqrt(2) * wc
        k2 = wc * wc
        den = k2 + k1 + 1
        
        self.b = np.array([k2/den, 2*k2/den, k2/den])
        self.a = np.array([1.0, 2*(k2-1)/den, (k2-k1+1)/den])

        # State buffers for each joint (assuming 6 joints)
        self.z1 = np.zeros(6)
        self.z2 = np.zeros(6)

    def filter_data(self, x):
        """Processes a vector of 6 joint velocities through a 2nd order Butterworth filter"""
        # Difference equation implementation (Direct Form II)
        out = self.b[0]*x + self.z1
        self.z1 = self.b[1]*x - self.a[1]*out + self.z2
        self.z2 = self.b[2]*x - self.a[2]*out
        return out