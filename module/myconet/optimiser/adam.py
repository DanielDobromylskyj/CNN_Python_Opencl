import numpy as np


class ADAM:  # Adaptive Momentum Estimation
    def __init__(self, shape, beta1=0.9, beta2=0.999):
        """ Adaptive Momentum Estimation (Algorithm) """
        self.beta_1 = beta1  # First moment decay
        self.beta_2 = beta2  # Second moment decay
        self.epsilon = 1e-7  # Small constant to avoid division by zero
        self.time_step = 0

        self.m = np.zeros(shape, dtype=np.float32)  # First moment (m_t)
        self.v = np.zeros(shape, dtype=np.float32)  # Second moment (v_t)

    def optimise(self, parameters, gradients):
        """ Performs calculations to controll the paramters of the network"""
        gradients = gradients.get_as_array()

        for param in range(len(parameters)):
            g_t = gradients[param]  # Gradient for parameter at time t

            # Update biased first moment estimate (m_t)
            self.m[param] = self.beta_1 * self.m[param] + (1 - self.beta_1) * g_t

            # Update biased second moment estimate (v_t)
            self.v[param] = self.beta_2 * self.v[param] + (1 - self.beta_2) * (g_t ** 2)

            # Compute bias-corrected estimates
            m_hat = self.m[param] / (1 - self.beta_1 ** self.time_step)
            v_hat = self.v[param] / (1 - self.beta_2 ** self.time_step)

            # Update parameter using ADAM formula
            parameters[param] += m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters

    def step(self):
        self.time_step += 1
