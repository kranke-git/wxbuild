# kranke - January 2026
# Script to define a set of CMIP6-related utilities

import numpy  as np
import xarray as xr
from   physicsutils  import dpt2q, q2dpt

def CalcGlobalDT( cmipdir, model, member, histperiod, futperiod, futexp ):
    """_summary_

    Args:
        cmipdir (_type_): _description_
        model (_type_): _description_
        member (_type_): _description_
        histperiod (_type_): _description_
        futperiod (_type_): _description_
        futexp (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Figure out deltaTG
    # If the year is past 2014 for historical, use ssp245
    if histperiod[ 0 ] > 2014:
        histfileG    = f"{cmipdir}/ssp245_Concatenated_tas_{model}_{member}_Amon_GlobalAverages_AnnualAverages.nc"
    else:
        histfileG    = f"{cmipdir}/historical_Concatenated_tas_{model}_{member}_Amon_GlobalAverages_AnnualAverages.nc"
    # Future file
    futfileG     = f"{cmipdir}/{futexp}_Concatenated_tas_{model}_{member}_Amon_GlobalAverages_AnnualAverages.nc"
    # Extract the tas values for the historical and future periods
    histVal      = xr.open_dataset( histfileG )['tas'].sel( time = slice( f"{histperiod[0]}-01-01", f"{histperiod[-1]}-12-31" ) ).mean( dim="time" ).values[0,0]
    if np.isnan( histVal ):
        raise ValueError(f"Historical data for {model} is not available for the specified period {histperiod}.")
    futVal       = xr.open_dataset( futfileG )['tas'].sel( time = slice( f"{futperiod[0]}-01-01", f"{futperiod[-1]}-12-31" ) ).mean( dim="time" ).values[0,0]
    # Calculate the delta global temperature
    deltaTG      = futVal - histVal
    return deltaTG

def getPatternCoefficients( cmipdir, experiments, realization, grid, month, locCoords ):
    coefs     = {}
    variables       = [ 'tas', 'uas', 'vas', 'ps', 'huss', 'tasmax', 'tasmin' ] 
    for variable in variables:
        n4file  = f"{cmipdir}/PatternScalingCoefficients_{variable}_{experiments}_{realization}_{grid}_M{month}.nc"
        pscoef  = xr.open_dataset( n4file )['slope'].sel(**locCoords, method = 'nearest' ).values
        coefs[ variable ] = pscoef
    return coefs

def calculateShift( coefs, deltaTG, currentPres, currentDpt, tmy3M ):
    avgShift           = {}
    avgShift[ 'dbt' ]  = coefs[ 'tas']  * deltaTG
    avgShift[ 'pres' ] = coefs[ 'ps' ]  * deltaTG
    avgShift[ 'u' ]    = coefs[ 'uas' ] * deltaTG
    avgShift[ 'v' ]    = coefs[ 'vas' ] * deltaTG
            
    # Figure out dewpoint
    avgP      = currentPres
    avgDPT    = currentDpt
    avgQ      = dpt2q( avgDPT , avgP / 100 )
    newQ      = avgQ + coefs['huss']*deltaTG
    newP      = avgP + coefs['ps']*deltaTG
    newDPT    = q2dpt( newQ, newP / 100 )
    avgShift[ 'dpt' ] = newDPT - avgDPT
    
    # diurnal temperature range
    dtasmax = deltaTG * coefs['tasmax']
    dtasmin = deltaTG * coefs['tasmin']
    daily_max          = tmy3M.groupby('date')[ 'dbt' ].max().reset_index()
    dbtmax_avg         = daily_max['dbt'].mean()
    daily_min          = tmy3M.groupby('date')[ 'dbt' ].min().reset_index()
    dbtmin_avg         = daily_min['dbt'].mean()

    avgShift[ 'at' ] = ( dtasmax - dtasmin ) / ( dbtmax_avg - dbtmin_avg )

    return avgShift
