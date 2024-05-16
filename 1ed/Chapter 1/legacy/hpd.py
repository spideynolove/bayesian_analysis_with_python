# from __future__ import division
# print(8/7)
# print(8//7)
import numpy as np
import scipy.stats.kde as kde


def hpd_grid(sample, alpha=0.05, roundto=2):
    '''Calculate highest posterior density
    - Posterior probability (Xác suất hậu nghiệm)
    - HPD is the minimum width Bayesian credible interval (Khoảng tin cậy Bayes)
    - unimodal | biomodal | multimodal distributions

    Parameters
    ----------
    sample: An array containing MCMC samples
    alpha: Desired probability of type I error (Sai lầm loại I)
    roundto: Number of digits after the decimal point for the results

    Returns
    ----------
    hpd: array with the lower 
    '''
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]

    # lower and upper bound
    l, u = np.min(sample), np.max(sample)
    
    spaced_samples = 2000
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, spaced_samples)
    y = density.evaluate(x)

    xy = sorted(zip(x, y/np.sum(y)), key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0

    hdv = []    # highest posterior value
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-1)/20
    hpd = []    # highest posterior density
    hpd.append(round(min(hdv), roundto))

    for i in range(1, len(hdv)):
        if hdv[i] - hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd - list(zip(ite, ite))
    modes = []

    for value in hpd:
        x_hpd = x[(x > value[0]) & (x < value[1])]
        y_hpd = x[(x > value[0]) & (x < value[1])]
        modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes