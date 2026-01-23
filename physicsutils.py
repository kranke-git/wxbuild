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
