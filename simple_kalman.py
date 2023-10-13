import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

landmark = 0.0

def motion_update(mu, sigma):
    motion_noise = 0.1
    mu_next = mu - 1
    sigma_next = sigma + motion_noise
    return (mu_next, sigma_next)

def observation_update(mu, sigma, measurement):
    # z = 3.0*x.0 + measurement_noise
    C = 3.0
    measurement_noise = 2.0
    K = sigma*C/(C*sigma*C + measurement_noise)
    mu_next = mu + K*(measurement - C*mu)
    sigma_next = (1.0 - K*C)*sigma

    return (mu_next, sigma_next)

def kalman_filter(mu, sigma, measurement):
    mu, sigma = motion_update(mu, sigma)
    mu, sigma = observation_update(mu, sigma, measurement)
    return (mu, sigma)

def plot_x(mu, sigma):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), "g")

    plt.axvline(x=[landmark])
    plt.show()

def run():
    observations = [30.1, 27.2, 24.1, 21.3, 19.0, 15.8, 12.1, 9.9]

    mu = 5.0
    sigma = 5.0

    plot_x(mu, sigma)
    print(mu, sigma)

    for obs in observations:
        mu, sigma = kalman_filter(mu, sigma, obs)
        print(mu, sigma)
        plot_x(mu, sigma)

def main():
    run()

if __name__ == "__main__":
    main()