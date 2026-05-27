# kranke - May 2026
# Script to define useful arma utility functions for fitting ARMA models to epw data from the main class


import scipy.stats                  as stats
import statsmodels.api              as sm
import numpy                        as np
from   scipy.optimize               import curve_fit
from   statsmodels.stats.diagnostic import acorr_ljungbox
from   sklearn.linear_model         import LinearRegression
import warnings

def cos_model(hour, A, phi, c):
    """
    Cosine model for fitting diurnal cycles.
    Parameters
    ----------
    hour : array-like
        Array of hours (0-23).
    A : float
        Amplitude of the cosine function.
    phi : float
        Phase shift of the cosine function.
    c : float
        Vertical shift of the cosine function.
    Returns
    -------
    array-like
        Values of the cosine function for the given hours.
    """
    return A * np.cos(2 * np.pi * hour / 24 + phi) + c


def fitDiurnalCycle( tmy3, month, variable ):
    """
    Fit a cosine model to the diurnal cycle of a specified variable for a given month.
    Parameters
    ----------
    tmy3 : DataFrame
        DataFrame containing TMY3 data.
    month : int
        Month for which to fit the diurnal cycle (1-12).
    variable : str
        Variable to fit (e.g., 'dbt', 'rh', 'pres').
    Returns
    -------
    tuple
        Tuple containing the fitted diurnal cycle values, amplitude, phase shift, and vertical shift.
    """
    
    typicalDbt = tmy3[ tmy3[ 'Month' ] == month ].groupby('Hour')[ variable ].mean().reset_index()
    # Fit the model
    popt, _   = curve_fit(cos_model, typicalDbt['Hour'], typicalDbt[ variable ], p0=[1, 0, 0]) 
    A, phi, c = popt    
    dbtCycle  = cos_model( tmy3[ 'Hour' ][ tmy3[ 'Month' ] == month ], A, phi, c )
    return dbtCycle, A, phi, c

def checkResiduals( residuals ):
    """
    Check if Residuals from a model are white noise (uncorrelated and normal)    
    """
    
    # Ljungs is for correlation, shapiro for normality
    ljungs  = acorr_ljungbox( residuals, lags = 20 ) # Test correlations
    shapiro = stats.shapiro( residuals )
    #if ( ljungs['lb_pvalue'] < 0.01).any():
    #    warnings.warn( "ljungs test have lags with p-val < 0.01", category=None, stacklevel=1)
    # Plot normality test with different distributions
    #fig, ax = plt.subplots(1, 3, figsize=(15, 5))    
    #stats.probplot( residuals, dist="norm", plot=ax[0])
    #paramsT = stats.t.fit( residuals )
    #stats.probplot( residuals, dist="t", sparams=paramsT, plot=ax[1] )  # pass the degrees of freedom
    #params_gamma = stats.gamma.fit( residuals )
    #stats.probplot( residuals, dist="gamma", sparams=params_gamma, plot=ax[2])
    
    # Return
    return ljungs, shapiro

def RegressArma( variable, exog, armaOrd ):    
    """
    Model any met variable as a linear regression to dbt + ARMA(armaOrd) for residuals
    This is useful if variables are correlated
    """
    
    # Get linear regression part in
    variable.index = variable.index - variable.index[ 0 ] # Reset index to start from zero
    correlation    = np.corrcoef( exog, variable )[0, 1]
    modelVar       = LinearRegression()
    modelVar.fit( exog.values.reshape(-1, 1), variable.values )
    slope          = modelVar.coef_
    intercept      = modelVar.intercept_

    # Fit ARMA Model and calculate the residuals
    residualsF   = variable - modelVar.predict( exog.values.reshape(-1,1) )
    with warnings.catch_warnings():
    
        warnings.filterwarnings(
            "ignore",
            message="Non-stationary starting autoregressive parameters"
        )

        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting MA parameters"
        )
        
        warnings.filterwarnings(
            "ignore",
            message="Maximum Likelihood optimization failed to converge"
        )


        armaVarModel = sm.tsa.ARIMA( residualsF, order = armaOrd, trend = 'n' )   
        armaVarFit   = armaVarModel.fit()
        armaRes      = armaVarFit.resid.dropna( )
    
    # Returns 
    return correlation, slope, intercept, modelVar, armaVarFit, armaRes


