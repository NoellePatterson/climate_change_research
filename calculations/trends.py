import glob
import os
import pandas as pd
import numpy as np
import pymannkendall as mk
import statsmodels.api as sm
import statsmodels.formula.api as smf

'''
Calculate Mann-Kendall trends in FFC historic reference streamflow data
'''

def calc_mk_trend(ffc_data, results_dicts):
    counter = 0
    for gage_index, gage in enumerate(ffc_data):
        results_dicts[gage_index]['results']['dw_stat'] = np.nan
        for index, value in enumerate(gage['ffc_metrics'].index):
            metric = gage['ffc_metrics'].loc[value]
            # print('testing '+ value)
            # drop rows with 'None' string as value
            metric = metric.mask(metric.eq('None')).dropna()
            # only perform trend analyses if years of data are above ten
            if len(metric) < 11:
                break
            # convert string numbers to floats
            for i, val in enumerate(metric):
                metric[i] = float(val)
            x = []
            for index in metric.index:
                x.append(int(index))
            # statmodels analysis requires column of constants for matrix multiplication
            x = sm.add_constant(x)
            y = np.asarray(metric)
            # Calculate dw for test of autocorrelation
            dw_stat = durbin_watson(y,x)
            # if dw_stat < 1.2:
            #     print('check for correlation in ' + value)
            metric = pd.to_numeric(metric, errors='coerce')
            mk_stats, ljung = mk_and_ljung(metric)
            
            # if p-val insig, can report results and be done with the metric. Otherwise, need to remove autocorrelation.
            if float(ljung['lb_pvalue']) < 0.05:
                for lag in range(1,len(metric)):
                    if float(ljung['lb_pvalue']) < 0.05:
                        diff = differencing(metric, lag)
                        if len(diff) < 11:
                            print('no adjustment possible')
                            break
                        mk_stats, ljung = mk_and_ljung(diff)
                        if float(ljung['lb_pvalue']) < 0.05:
                            continue
                        else:
                            break
            else: 
                print('report results')
                # results_dicts[gage_index]['results']['dw_stat'] = dw_stat
                # results_dicts['mk_trend'] == mk   

    import pdb; pdb.set_trace()
    return results_dicts

def durbin_watson(y,x, axis=0):
    """
    adapted from statsmodels.stats.stattools
    Calculates the Durbin-Watson statistic
    Parameters
    ----------
    resids : array_like
    Returns
    -------
    dw : float, array_like
        The Durbin-Watson statistic.
    Notes
    -----
    The null hypothesis of the test is that there is no serial correlation.
    The Durbin-Watson test statistics is defined as:

       \sum_{t=2}^T((e_t - e_{t-1})^2)/\sum_{t=1}^Te_t^2

    The test statistic is approximately equal to 2*(1-r) where ``r`` is the
    sample autocorrelation of the residuals. Thus, for r == 0, indicating no
    serial correlation, the test statistic equals 2. This statistic will
    always be between 0 and 4. The closer to 0 the statistic, the more
    evidence for positive serial correlation. The closer to 4, the more
    evidence for negative serial correlation.
    For my data, a value above ~0.9-1.2 is considered Not autocorrelated (see DW significance tables)
    """

    model = sm.OLS(y,x).fit()
    Y_pred = model.predict(x)
    residuals = y-Y_pred

    diff_resids = np.diff(residuals, 1, axis=axis)
    if np.sum(residuals) == 0:
        dw = np.nan
    else:
        dw = np.sum(diff_resids**2, axis=axis) / np.sum(residuals**2, axis=axis)
    return dw

def mk_and_ljung(array):
    mk_stats = mk.original_test(array)
    x_vals = np.arange(1, len(array)+1, 1)
    y_vals = x_vals * mk_stats.slope + mk_stats.intercept
    residuals = array - y_vals
    ljung = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
    return mk_stats, ljung

def differencing(array, lag=1):
    diff = []
    for i in range(len(array)-lag):
        diff.append(array[i+lag] - array[i])
    return diff