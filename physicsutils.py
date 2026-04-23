# kranke - January 2026
# Script to define physics-related utility functions

import math

def dpt2q( dpt, pres ):
    """
    Function to convert dewpoint temperatures to specific humidity

    Args:
        dpt (float): dewpoint in Celsisu
        pres: mb or hPa
    """
    # Calculate the saturation vapor pressure at the dewpoint temperature
    e_s = 6.112 * math.exp((17.67 * dpt) / (dpt + 243.5) )
    
    # Calculate the actual vapor pressure (same as saturation vapor pressure at T_dew)
    e = e_s  # since at dewpoint, actual vapor pressure equals saturation vapor pressure
    
    # Calculate specific humidity from actual vapor pressure
    q = ( 0.622 * e ) / ( pres - (1-0.622)*e )  # Specific humidity in kg/kg
    
    return q

def q2dpt( q, P ):
    """
    Converts specific humidity to dewpoint temperature.

    Parameters:
    q (float): Specific humidity in kg/kg.
    P (float): Air pressure in hPa or mb.

    Returns:
    float: Dewpoint temperature in Celsius.
    """
    
    # Calculate the actual vapor pressure from specific humidity
    e = (q * P) / (0.622 + q)
    
    # Calculate dewpoint temperature from the actual vapor pressure
    dpt = (243.5 * math.log(e / 6.112)) / (17.67 - math.log(e / 6.112))
    
    return dpt

def dbt_dpt2rh( dbt, dpt ):
    """
    Function to compute relative humidity from dry bulb and dewpoint
    This is just dividing saturation humidity at dewpoint (dpt) by the saturation humidity at actual temperature (dbt)
    """
    rh = 100 * ( np.exp( ( 17.625 * dpt ) / ( 243.04 + dpt ) ) / np.exp( ( 17.625 * dbt )/( 243.04 + dbt ) ) )
    return rh

def wind2uv(speed, direction):
    """
    Convert wind speed and direction to u and v components.
    
    Parameters:
        speed (float or array-like): Wind speed (m/s).
        direction (float or array-like):    Wind direction (degrees), 
                                            where 0° is from the north, 90° is from the east.
    
    Returns:
        tuple: u (zonal wind component, m/s), v (meridional wind component, m/s)
    """
    direction_rad = np.radians(direction)  # Convert degrees to radians
    u = -speed * np.sin(direction_rad)     # Negative sign because wind direction is "from"
    v = -speed * np.cos(direction_rad)
    return u, v

import numpy as np

def uv2wind(u, v):
    """
    Convert u and v wind components to wind speed and meteorological wind direction.
    
    Parameters:
        u (float or array-like): Zonal wind component (m/s).
        v (float or array-like): Meridional wind component (m/s).
    
    Returns:
        tuple: 
            - speed (float or array-like): Wind speed (m/s).
            - direction (float or array-like): Wind direction (degrees, meteorological convention).
            Direction is "from" that angle, with 0° = North, 90° = East.
    """
    speed = np.hypot(u, v)  # Equivalent to sqrt(u^2 + v^2)
    direction = (270 - np.degrees(np.arctan2(v, u))) % 360  # Convert to meteorological convention
    
    return speed, direction

import numpy as np

def saturation_mixing_ratio(T, p):
    """
    Saturation mixing ratio (kg/kg)

    Parameters
    ----------
    T : float or array
        Temperature in Kelvin
    p : float or array
        Pressure in hPa

    Returns
    -------
    qs : float or array
        Saturation mixing ratio (kg/kg)
    """
    # saturation vapor pressure (hPa)
    T_C = T - 273.15
    es  = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    epsilon = 0.622
    qs = epsilon * es / (p - es)

    return qs



def es(T):
    """Saturation vapor pressure (hPa), Bolton formula"""
    T_C = T - 273.15
    return 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))


def des_dT(T):
    """
    d(es)/dT using Clausius-Clapeyron (hPa/K)
    """
    Lv = 2.5e6      # J/kg
    Rv = 461.5      # J/(kg K)
    e = es(T)
    return (Lv / (Rv * T**2)) * e

def dqsat_dT(T, p):
    """
    Derivative of saturation mixing ratio w.r.t temperature

    Parameters
    ----------
    T : float or array (K)
    p : float or array (hPa)

    Returns
    -------
    dqs_dT : kg/kg/K
    """
    epsilon = 0.622

    e = es(T)
    de_dT = des_dT(T)

    return epsilon * p * de_dT / (p - e)**2

import numpy as np

def rh_t2dpt( Tc, RH ):
    """
    Compute dew point temperature (°C) from dry bulb temperature (°C) and relative humidity (%).

    Parameters:
        Tc : temperature in °C
        RH : relative humidity in % (0-100)

    Returns:
        Td : dew point in °C
    """
    # Magnus constants (over water)
    a = 17.67
    b = 243.5  # °C

    gamma = np.log(RH / 100.0) + (a * Tc) / (b + Tc)
    Td_c = (b * gamma) / (a - gamma)

    return Td_c