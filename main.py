# kranke - January 2026
# Script to test out the class defined in epwclass.py

from   epwclass import epw_collection
import numpy as np

current_tmy = epw_collection( filetype = 'tmy', location = 'Boston__MA__USA' )
# current_tmy.downloadCmip( model = 'CanESM5' )
# future_tmy  = current_tmy.files[0].with_futureShift(    cmipdir = f"{current_tmy.data_directory}/cmip6", 
#                                                         params = {'model':'CanESM5'},
#                                                         savedir = None )

# future_amy_coll = current_amy.with_futureShifts( params = { 'model':'CanESM5', 'futyear': 2050, 'futexp':'ssp585'  }, saveflag = True )

# Test on the get anomalies function
df = current_tmy.getVariableAnomalies( params = { 'variable':'dbt', 'years': np.arange( 1950, 2101 ), 'futexp':'ssp585' } )