import warnings
import numpy as np
import pymc3 as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
palette = 'muted'
sns.set_palette(palette)
sns.set_color_codes(palette)
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

warnings.filterwarnings('ignore')

new_warnings = ''' 
# https://stackoverflow.com/questions/30798447/porting-pymc2-code-to-pymc3-hierarchical-model-for-sports-analytics
'''

data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45,
                 52.34, 55.65, 51.49, 51.86, 63.43, 53.00, 56.09,
                 51.93, 52.31, 52.33, 57.48, 57.44, 55.14, 53.93,
                 54.62, 56.09, 68.58, 51.36, 55.47, 50.73, 51.94,
                 54.95, 50.39, 52.91, 51.5, 52.68, 47.72, 49.73,
                 51.82, 54.99, 52.84, 53.19, 54.52, 51.46, 53.73,
                 51.61, 49.81, 52.42, 54.3, 53.84, 53.16])


def join_params():
    ''' Nuisance parameters and marginalized distributions '''
    np.random.seed(123)

    x = np.random.gamma(2, 1, 1000)
    y = np.random.normal(0, 1, 1000)

    data = pd.DataFrame(data=np.array([x, y]).T, columns=[
                        '$\\theta_1$', '$\\theta_2$'])
    sns.jointplot(x='$\\theta_1$', y='$\\theta_2$', data=data)

    plt.show()


def gaussian():
    ''' Gaussian inferences '''

    # sns.kdeplot(data)

    # remove outliers using the interquartile rule
    quant = np.percentile(data, [25, 75])
    iqr = quant[1] - quant[0]
    upper_b = quant[1] + iqr * 1.5
    lower_b = quant[0] - iqr * 1.5

    clean_data = data[(data > lower_b) & (data < upper_b)]

    print(np.mean(data), np.std(data))
    print(np.mean(clean_data), np.std(clean_data))

    sns.kdeplot(clean_data, color='g')

    plt.xlabel('$x$', fontsize=16)

    with pm.Model() as model_g:
        mu = pm.Uniform('mu', lower=40, upper=70)
        sigma = pm.HalfNormal('sigma', sd=10)
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data)
        step = pm.NUTS()
        trace_g = pm.sample(1000, step=step)

    chain_g = trace_g[100:]
    
    # az.plot_trace(chain_g)

    # print(pm.summary(chain_g))

    y_pred = pm.sample_posterior_predictive(
        chain_g, 100, model_g, size=len(data))

    sns.kdeplot(data, color='b')

    for i in y_pred['y']:
        # print(i.shape)
        dataset = pd.DataFrame({'Column1': i[:, 0], 'Column2': i[:, 1]})
        # print(dataset)
        sns.kdeplot(dataset.Column2, color='r', alpha=0.1)

    plt.title('Gaussian model', fontsize=16)
    plt.xlabel('$x$', fontsize=16)

    plt.show()


def gauss_robust():
    plt.figure(figsize=(8, 6))
    x_values = np.linspace(-10, 10, 200)
    for df in [1, 2, 5, 30]:
        distri = stats.t(df)
        x_pdf = distri.pdf(x_values)
        plt.plot(x_values, x_pdf, label=r'$\nu$ = {}'.format(df))

    x_pdf = stats.norm.pdf(x_values)

    plt.plot(x_values, x_pdf, label=r'$\nu = \infty$')

    plt.xlabel('$x$', fontsize=16)
    plt.ylabel('$pdf(x)$', fontsize=16, rotation=90)
    plt.legend(loc=0, fontsize=14)
    plt.xlim(-7, 7)

    plt.show()


def t_dist():
    with pm.Model() as model_t:
        mu = pm.Uniform('mu', 40, 75)
        sigma = pm.HalfNormal('sigma', sd=10)
        nu = pm.Exponential('nu', 1/30)
        y = pm.StudentT('y', mu=mu, sd=sigma, nu=nu, observed=data)
        step = pm.NUTS()
        trace_t = pm.sample(1000, step=step)

    chain_t = trace_t[100:]

    # az.plot_trace(chain_t)

    print(pm.summary(chain_t))

    y_pred = pm.sample_posterior_predictive(
        chain_t, 100, model_t, size=len(data))
    sns.kdeplot(data, c='b')

    for i in y_pred['y']:
        dataset = pd.DataFrame({'Column1': i[:, 0], 'Column2': i[:, 1]})
        sns.kdeplot(dataset.Column2, color='r', alpha=0.1)
        pass

    plt.xlim(35, 75)
    plt.title("Student's t model", fontsize=16)
    plt.xlabel('$x$', fontsize=16)

    plt.show()


def tips_exam():
    pass


def main():
    # join_params()
    gaussian()
    # gauss_robust()
    # t_dist()
    pass


if __name__ == "__main__":
    main()
