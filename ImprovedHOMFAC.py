import numpy as np
from scipy.ndimage import shift


class IHOMFAC:
    """
    `Implements the Improved High Order Model Free Adaptive Control (IHOMFAC) algorithm.

    Attributes:
    - eta: A float representing the learning rate for the control input.
    - lam: A float representing the forgetting factor for the control input.
    - mu: A float representing the forgetting factor for the parameter estimate.
    - rho: A float representing the weight for the error term in the control input.
    - eps: A float representing the tolerance for small values.
    - phi_init: A float representing the initial value for the parameter estimate.
    - y_init: A float representing the initial value for the output measurement.
    - u_init: A float representing the initial value for the control input.
    - alpha: A list or array of floats representing the coefficients for the control input.
    - beta: A list or array of floats representing the coefficients for the parameter estimate.

    Methods:
    - update_y: Updates the y array with the latest measurement.
    - estimate_phi: Estimates the value of phi based on the current state.
    - calculate_control_input: Calculates the control input based on the current state and setpoint.
    - run_iteration: Runs a single iteration of the IHOMFAC algorithm.`
    """

    def __init__(self, **kwargs):
        """
        Initializes an instance of the Improved High Order Model Free Adaptive Control (IHOMFAC) class.

        Args:
        - **kwargs: A dictionary of keyword arguments that can be passed to the constructor.
            - eta: A float representing the learning rate for the control input. Default is 0.8.
            - lam: A float representing the forgetting factor for the control input. Default is 0.1.
            - mu: A float representing the forgetting factor for the parameter estimate. Default is 0.01.
            - rho: A float representing the weight for the error term in the control input. Default is 0.8.
            - eps: A float representing the tolerance for small values. Default is 5e-16.
            - phi_init: A float representing the initial value for the parameter estimate. Default is 0.05.
            - y_init: A float representing the initial value for the output measurement. Default is 0.
            - u_init: A float representing the initial value for the control input. Default is 0.
            - alpha: A list or array of floats representing the coefficients for the control input. Default is an empty list.
            - beta: A list or array of floats representing the coefficients for the parameter estimate. Default is an empty list.
        """
        self.eta = kwargs.get("eta", 0.8)
        self.lam = kwargs.get("lam", 0.1)
        self.mu = kwargs.get("mu", 0.01)
        self.rho = kwargs.get("rho", 0.8)
        self.eps = kwargs.get("eps", 5e-16)
        phi_init = kwargs.get("phi_init", 0.05)
        y_init = kwargs.get("y_init", 0)
        u_init = kwargs.get("u_init", 0)
        alpha = np.array(kwargs.get("alpha", []))
        beta = np.array(kwargs.get("beta", []))

        self.alpha = np.array(alpha, dtype=float).reshape(1, -1)
        self.beta = np.array(beta, dtype=float).reshape(1, -1)

        self.y = np.array([float(y_init)] * 2)
        self.u = np.array([float(u_init)] * (len(alpha))).reshape(-1, 1)
        print(self.u.shape)
        self.phi_init = phi_init
        self.phi_init_sign = np.sign(phi_init)
        self.phi = np.zeros((len(beta), 1))
        self.phi[0, 0] = phi_init

    def update_y(self, y_meas):
        updated_y = shift(
            self.y, 1, cval=y_meas
        )  # Shifts the y array to update the measurements
        updated_y[np.abs(updated_y) < self.eps] = 0
        self.y = updated_y

    def estimate_phi(self):
        sum_beta_phi = self.beta @ self.phi
        delta_ut_1 = self.u[1] - self.u[2]  # delta(u(t-1)) = u(t-1) - u(t-2)
        delta_yt = self.y[0] - self.y[1]  # delta(y(t)) = y(t) - y(t-1)

        phi_est = sum_beta_phi + (self.eta * delta_ut_1) * (
            delta_yt - delta_ut_1 * sum_beta_phi
        ) / (self.mu + delta_ut_1**2)

        if (
            (phi_est <= self.eps)
            or (abs(delta_ut_1) <= self.eps)
            or (np.sign(phi_est) != self.phi_init_sign)
        ):
            phi_est = self.phi_init

        updated_phi = shift(
            self.phi, (1, 0), cval=phi_est
        )  # Shifts the phi array to update the estimate
        updated_phi[np.abs(updated_phi) < self.eps] = 0
        self.phi = updated_phi

    def calculate_control_input(self, y_setpoint):
        sum_alpha = self.alpha @ self.u
        coeff_denom = self.lam + self.phi[0] ** 2
        error = y_setpoint - self.y[0]

        control_input = (
            (self.phi[0] ** 2) * self.u[1]
            + self.lam * sum_alpha
            + self.rho * self.phi[0] * error
        ) / coeff_denom
        updated_u = shift(
            self.u, (1, 0), cval=control_input
        )  # Shifts the u array to update the control input
        updated_u[np.abs(updated_u) < self.eps] = 0
        self.u = updated_u

    def run_iteration(self, y_meas, y_setpoint):
        self.update_y(y_meas)
        self.estimate_phi()
        self.calculate_control_input(y_setpoint)
