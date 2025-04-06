import numpy as np
import scipy.stats


def sine_wave(x, A, f, phi, C):
    """
    Generate a sine wave.

    Parameters
    ----------
    x : array-like or float
        Timestep(s) at which to evaluate the sine wave.
    A : float
        Amplitude of the sine wave.
    f : float
        Frequency of the sine wave (in cycles per unit of x).
    phi : float
        Phase shift of the sine wave (in radians).
    C : float
        Vertical offset (baseline) of the sine wave.

    Returns
    -------
    float or array-like
        Value(s) of the sine wave at the given timestep(s).
    """
    return A * np.sin(2 * np.pi * f * x + phi) + C


def find_gamma_parameters(
    p_target, q_target, alpha_range, tau_bar_range, tolerance=1e-4, step_size=0.1
):
    """
    Find shape (alpha) and mean (tau_bar) parameters of a gamma distribution
    that match a target PDF and CDF value at the same quantile, to generate an arbitrary tail.

    Parameters
    ----------
    p_target : float
        Target value of the probability density function (PDF) at a specific quantile.
    q_target : float
        Target value of the cumulative distribution function (CDF) (i.e., quantile level).
    alpha_range : tuple of float
        Range of shape parameters (alpha) to search, as (min_alpha, max_alpha).
    tau_bar_range : tuple of float
        Range of mean transit times (tau_bar, in years) to search, as (min_tau_bar, max_tau_bar).
    tolerance : float, optional
        Acceptable error between target PDF value and computed PDF, by default 1e-4.
    step_size : float, optional
        Step size for scanning the parameter space, by default 0.1.

    Returns
    -------
    list of tuple
        List of tuples (alpha, tau_bar) where the PDF at the quantile corresponding 
        to `q_target` matches `p_target` within the specified `tolerance`. `tau_bar` is returned in days.
    """
    matching_params = []

    for alpha in np.arange(alpha_range[0], alpha_range[1] + step_size, step_size):
        for tau_bar in np.arange(tau_bar_range[0], tau_bar_range[1] + step_size, step_size) * 365.25:
            # Find the quantile (x_q) for the given CDF value q_target
            x_q = scipy.stats.gamma.ppf(q_target, a=alpha, scale=tau_bar / alpha)

            # Calculate the PDF value at this quantile
            pdf_value = scipy.stats.gamma.pdf(x_q, a=alpha, scale=tau_bar / alpha)

            # Check if the PDF value matches the target within the given tolerance
            if np.abs(pdf_value - p_target) <= tolerance:
                matching_params.append((alpha, tau_bar))

    return matching_params


def find_nearest_greater(array, value):
    """
    Find the index of the nearest value in the array that is strictly greater than a given value.

    Parameters
    ----------
    array : array-like
        Input array to search through.
    value : float or int
        Reference value to compare against.

    Returns
    -------
    int
        Index of the nearest value in the array that is strictly greater than `value`.

    Raises
    ------
    ValueError
        If no value in the array is greater than the given `value`.
    """
    array = np.asarray(array)
    greater_values = array[array > value]

    if greater_values.size == 0:
        raise ValueError("No element in the array is greater than the provided value.")

    nearest_value = greater_values.min()
    idx = np.where(array == nearest_value)[0][0]

    return idx
