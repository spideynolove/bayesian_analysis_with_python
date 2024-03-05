import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import warnings

warnings.filterwarnings('ignore')

palette = 'muted'
sns.set_palette(palette)
sns.set_color_codes(palette)


def posterior_grid(grid_points=100, heads=6, tosses=9):
    """
    A grid implementation for the coin-flip problem
    """
    # define a grid
    grid = np.linspace(0, 1, grid_points)

    # define prior
    prior = np.repeat(5, grid_points)  # uniform? why 5: reasonable interval

    # prior = (grid  <= 0.4).astype(int)  # truncated
    # prior = abs(grid - 0.5)  # "M" prior

    # compute likelihood at each point in the grid
    likelihood = stats.binom.pmf(heads, tosses, grid)

    # compute product of likelihood and prior
    unstd_posterior = likelihood * prior

    # print(likelihood.shape, prior.shape, unstd_posterior.shape)

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior / unstd_posterior.sum()
    return grid, posterior


def coin_flip():
    points = 15
    h, n = 1, 4

    grid, posterior = posterior_grid(points, h, n)

    plt.plot(grid, posterior, 'o-')
    plt.plot(0, 0, label='heads = {}\ntosses = {}'.format(h, n), alpha=0)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.legend(loc=0, fontsize=14)
    plt.show()


def estimate_pi():
    N = 10000
    # N = 10

    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    # print(inside.sum())

    pi = inside.sum()*4/N
    error = abs((pi - np.pi)/pi) * 100
    outside = np.invert(inside)

    # '''
    plt.plot(x[inside], y[inside], 'b.')
    plt.plot(x[outside], y[outside], 'r.')  # red - square

    plt.plot(0, 0, label='$\hat \pi$ = {:4.3f}\nerror = {:4.3f}%'.format(
        pi, error), alpha=0)

    plt.axis('square')
    plt.legend(frameon=True, framealpha=0.9, fontsize=16)
    plt.show()
    # '''


def metropolis(func, steps=10000):
    """A very simple Metropolis implementation"""
    samples = np.zeros(steps)
    old_x = func.mean()
    old_prob = func.pdf(old_x)

    for i in range(steps):
        new_x = old_x + np.random.normal(0, 1)
        new_prob = func.pdf(new_x)

        acceptance = new_prob/old_prob
        if acceptance >= np.random.random():
            samples[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            samples[i] = old_x
    return samples


def computing_metropolis():
    np.random.seed(345)

    func = stats.beta(0.4, 2)
    samples = metropolis(func=func)

    x = np.linspace(0.01, .99, 100)
    y = func.pdf(x)

    plt.xlim(0, 1)
    plt.plot(x, y, 'r-', lw=3, label='True distribution')

    plt.hist(samples, bins=30, density=True,
             label='Estimated distribution', stacked=True)
    plt.xlabel('$x$', fontsize=14)
    plt.ylabel('$pdf(x)$', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()


def coin_flip_pymc3():
    np.random.seed(123)
    n_experiments = 4   # 40
    theta_real = 0.35  # unkwon value in a real experiment
    
    data = stats.bernoulli.rvs(p=theta_real, size=n_experiments)
    # print(data)

    with pm.Model() as our_first_model:
        # a priori
        theta = pm.Beta('theta', alpha=1, beta=1)
        # likelihood
        y = pm.Bernoulli('y', p=theta, observed=data)
        #y = pm.Binomial('theta',n=n_experimentos, p=theta, observed=sum(datos))
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(1000, step=step, start=start)

    burnin = 0  # no burnin
    chain = trace[burnin:]
    pm.traceplot(chain, lines={'theta': theta_real})
    plt.show()

    with our_first_model:
        step = pm.Metropolis()
        multi_trace = pm.sample(1000, step=step, cores=4)

    burnin = 0  # no burnin
    multi_chain = multi_trace[burnin:]
    pm.traceplot(multi_chain, lines={'theta': theta_real})
    plt.show()

    # print(pm.gelman_rubin(multi_chain))

    pm.forestplot(multi_chain, var_names=['theta'])
    plt.show()

    print(pm.summary(multi_chain))
    
    # print(pm.df_summary(multi_chain))

    pm.autocorrplot(chain)
    plt.show()

    # print(pm.effective_n(multi_chain)['theta'])
    
    az.plot_posterior(chain, kind='kde')
    plt.show()

    az.plot_posterior(chain, kind='kde', rope=[0.45, .55])
    plt.show()

    az.plot_posterior(chain, kind='kde', ref_val=0.5)
    plt.show()


def main():
    # coin_flip()
    # estimate_pi()
    # computing_metropolis()
    coin_flip_pymc3()


if __name__ == "__main__":
    sample = '''
    https://stackoverflow.com/questions/30798447/porting-pymc2-code-to-pymc3-hierarchical-model-for-sports-analytics
    '''
    main()
