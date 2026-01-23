# kranke - January 2026
# Miscellaneous utility functions

import numpy        as np
from   physicsutils import wind2uv, uv2wind, dbt_dpt2rh

def shift_tuple( t, new_center ):
    start, end = t
    half_range = (end - start) // 2
    return (new_center - half_range, new_center + half_range)

def swapMonthTmy( tmy3_mod, idxmonth, avgShift = {'dbt': 0, 'dpt': 0, 'pres':0, 'u':0, 'v':0}, swapYears = None ):
    """
        Function to assign all the emulated variables to the new dataframe
        Args:
            tmy3_mod (DataFrame): DataFrame containing the TMY3 data to be modified
            idxmonth (list): List of indices corresponding to the month to be modified
            avgShift (dict, optional): Dictionary containing the average shifts for each variable. Defaults to {'dbt': 0, 'dpt': 0, 'pres':0, 'u':0, 'v':0}.
            swapYears (list, optional): List of years to swap into the 'Year' column. Defaults to None.
        Returns:    
            DataFrame: Modified TMY3 DataFrame
    """
    
    # dbt, dpt, pres
    tmy3_mod.loc[ idxmonth, 'dbt'  ] += ( tmy3_mod.loc[ idxmonth, 'dbt'  ] -  tmy3_mod.loc[ idxmonth, 'dbt'  ].mean() ) * avgShift[ 'at' ] # Stretching
    tmy3_mod.loc[ idxmonth, 'dbt'  ] += round( avgShift[ 'dbt' ],  1 ) # Shifting
    tmy3_mod.loc[ idxmonth, 'dpt'  ] += round( avgShift[ 'dpt' ],  1 )
    tmy3_mod.loc[ idxmonth, 'pres' ] += round( avgShift[ 'pres' ], -2 )
    
        
    # Winds
    utmy, vtmy = wind2uv( tmy3_mod.loc[ idxmonth, 'wspd'  ], tmy3_mod.loc[ idxmonth, 'wdir'  ] )
    uShifted = utmy + avgShift[ 'u' ]
    vShifted = vtmy + avgShift[ 'v' ]
    wspd_shift, wdir_shift           = uv2wind( uShifted, vShifted )
    tmy3_mod.loc[ idxmonth, 'wspd' ] = round( wspd_shift, 1 )
    tmy3_mod.loc[ idxmonth, 'wdir' ] = round( wdir_shift, -1 )
    # relative humidity
    tmy3_mod.loc[ idxmonth, 'rh' ] = round( dbt_dpt2rh( tmy3_mod.loc[ idxmonth, 'dbt'  ], tmy3_mod.loc[ idxmonth, 'dpt' ] ), 0 )
    
    # If future file, swap the years as well (draw a random year in the future period, that is different from previous month)
    if swapYears is not None:
        if len( swapYears ) != 1:
            prevMonth     = tmy3_mod.loc[ idxmonth, 'Month'  ].unique()[0] - 1
            if prevMonth > 1:
                idxPrevMonth  = tmy3_mod.index[ tmy3_mod['Month'] == prevMonth].tolist() 
                prevMonthYear = tmy3_mod.loc[ idxPrevMonth, 'Year'  ].unique()[0]
                possibleYears = swapYears[ swapYears != prevMonthYear ]
            else:
                possibleYears = swapYears
                
            # Now draw the random year and assign
            randomYear    = np.random.choice( possibleYears, size=1, replace=True )[ 0 ]
            tmy3_mod.loc[ idxmonth, 'Year'] = randomYear
        else:
            tmy3_mod.loc[ idxmonth, 'Year'] = swapYears[ 0 ]

    return( tmy3_mod )
