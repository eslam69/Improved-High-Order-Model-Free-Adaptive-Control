import numpy as np
from scipy.ndimage import shift

class HOMFAC:
    """
    High Order Model Free Adaptive Control (HOMFAC) class.

    This class implements the HOMFAC algorithm for adaptive control of a system.

    Attributes:
        eta (float): Learning rate for the estimation of phi.
        lam (float): Forgetting factor for the control input calculation.
        mu (float): Regularization parameter for the estimation of phi.
        rho (float): Weight for the error term in the control input calculation.
        eps (float): Tolerance value for numerical stability.
        alpha (numpy.ndarray): Row vector of coefficients for the control input calculation.
        y (numpy.ndarray): Array of past measurements of the system output.
        u (numpy.ndarray): Column vector of past control inputs.
        phi_init (float): Initial value for the estimation of phi.
        phi_init_sign (float): Sign of the initial value for the estimation of phi.
        phi (numpy.ndarray): Array of past estimates of phi.

    Methods:
        update_y(y_meas): Updates the array of past measurements of the system output.
        estimate_phi(): Estimates the value of phi based on the past measurements and control inputs.
        calculate_control_input(y_setpoint): Calculates the control input based on the current and past measurements and control inputs.
        run_iteration(y_meas, y_setpoint): Runs one iteration of the HOMFAC algorithm.

    """
    def __init__(self, **kwargs):
        """
        Initializes the HOMFAC class with the specified parameters.

        Args:
            eta (float, optional): Learning rate for the estimation of phi. Defaults to 0.8.
            lam (float, optional): Forgetting factor for the control input calculation. Defaults to 0.1.
            mu (float, optional): Regularization parameter for the estimation of phi. Defaults to 0.01.
            rho (float, optional): Weight for the error term in the control input calculation. Defaults to 0.8.
            eps (float, optional): Tolerance value for numerical stability. Defaults to 5e-16.
            phi_init (float, optional): Initial value for the estimation of phi. Defaults to 0.01.
            y_init (float, optional): Initial value for the past measurements of the system output. Defaults to 0.
            u_init (float, optional): Initial value for the past control inputs. Defaults to 0.
            alpha (list or numpy.ndarray, optional): List or array of coefficients for the control input calculation. Defaults to [].

        Returns:
            None

        """
        self.eta = kwargs.get('eta', 0.8)
        self.lam = kwargs.get('lam', 0.1)
        self.mu = kwargs.get('mu', 0.01)
        self.rho = kwargs.get('rho', 0.8)
        self.eps = kwargs.get('eps', 5e-16)
        phi_init = kwargs.get('phi_init', 0.01)
        y_init = kwargs.get('y_init', 0)
        u_init = kwargs.get('u_init', 0)
        alpha = np.array(kwargs.get('alpha', []))

        self.alpha = np.array(alpha, dtype=float).reshape(1, -1) # row vec

        self.y = np.array([float(y_init)] * 2)
        self.u = np.array([float(u_init)] * (len(alpha))).reshape(-1, 1)   
        self.phi_init = phi_init
        self.phi_init_sign = np.sign(phi_init)
        self.phi = np.zeros((2, 1))
        self.phi[0,0] = phi_init
        print(self.phi)

    def estimate_phi(self):
        """
        Estimates the value of phi based on the past measurements and control inputs.

        Args:
            None

        Returns:
            None

        """
        delta_ut_1 = self.u[1] - self.u[2] #delta(u(t-1)) = u(t-1) - u(t-2)
        delta_yt = self.y[0] - self.y[1] #delta(y(t)) = y(t) - y(t-1)

        phi_est = self.phi[1,0] + (self.eta * delta_ut_1) * (delta_yt - delta_ut_1 * self.phi[1,0]) / (self.mu + delta_ut_1**2)

        if (phi_est <= self.eps) or (abs(delta_ut_1) <= self.eps) or (np.sign(phi_est) != self.phi_init_sign):
            phi_est = self.phi_init

        updated_phi = shift(self.phi, (1, 0), cval=phi_est) # Shifts the phi array to update the estimate
        updated_phi[np.abs(updated_phi) < self.eps] = 0
        self.phi = updated_phi

    def calculate_control_input(self, y_setpoint):
        """
        Calculates the control input based on the current and past measurements and control inputs.

        Args:
            y_setpoint (float): The setpoint for the system output.

        Returns:
            None

        """
        sum_alpha_u = self.alpha @ self.u
        denominator = self.lam + self.phi[0]**2
        error = y_setpoint - self.y[0]

        control_input = ((self.phi[0]**2) * self.u[1] + self.lam * sum_alpha_u + self.rho * self.phi[0] * error) / denominator
        updated_u = shift(self.u, (1, 0), cval=control_input) # Shifts the u array to update the control input
        updated_u[np.abs(updated_u) < self.eps] = 0
        self.u = updated_u

    def run_iteration(self, y_meas, y_setpoint):
        """
        Runs one iteration of the HOMFAC algorithm.

        Args:
            y_meas (float): The current measurement of the system output.
            y_setpoint (float): The setpoint for the system output.

        Returns:
            None

        """
        self.update_y(y_meas)

    def update_y(self, y_meas):
        """
        Updates the array of past measurements of the system output.

        Args:
            y_meas (float): The current measurement of the system output.

        Returns:
            None

        """
        updated_y = shift(self.y, 1, cval=y_meas) # Shifts the y array to update the measurements
        updated_y[np.abs(updated_y) < self.eps] = 0
        self.y = updated_y

    def estimate_phi(self):
        
        delta_ut_1 = self.u[1] - self.u[2] #delta(u(t-1)) = u(t-1) - u(t-2)
        delta_yt = self.y[0] - self.y[1] #delta(y(t)) = y(t) - y(t-1)

        phi_est = self.phi[1,0] + (self.eta * delta_ut_1) * (delta_yt - delta_ut_1 * self.phi[1,0]) / (self.mu + delta_ut_1**2)

        if (phi_est <= self.eps) or (abs(delta_ut_1) <= self.eps) or (np.sign(phi_est) != self.phi_init_sign):
            phi_est = self.phi_init

        updated_phi = shift(self.phi, (1, 0), cval=phi_est) # Shifts the phi array to update the estimate
        updated_phi[np.abs(updated_phi) < self.eps] = 0
        self.phi = updated_phi

    def calculate_control_input(self, y_setpoint):
        sum_alpha_u = self.alpha @ self.u
        denominator = self.lam + self.phi[0]**2
        error = y_setpoint - self.y[0]

        control_input = ((self.phi[0]**2) * self.u[1] + self.lam * sum_alpha_u + self.rho * self.phi[0] * error) / denominator
        updated_u = shift(self.u, (1, 0), cval=control_input) # Shifts the u array to update the control input
        updated_u[np.abs(updated_u) < self.eps] = 0
        self.u = updated_u

    def run_iteration(self, y_meas, y_setpoint):
        self.update_y(y_meas)
        self.estimate_phi()
        self.calculate_control_input(y_setpoint)
