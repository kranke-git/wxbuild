# kranke - January 2026
# Script to test out the class defined in epwclass.py

from epwclass import epw_collection

current_amy = epw_collection( filetype = 'amy', location = 'Boston__MA__USA' )
# current_tmy.downloadCmip( model = 'CanESM5' )
# future_tmy  = current_tmy.files[0].with_futureShift(    cmipdir = f"{current_tmy.data_directory}/cmip6", 
#                                                         params = {'model':'CanESM5'},
#                                                         savedir = None )

future_amy_coll = current_amy.with_futureShifts( params = { 'futyear': 2050  }, saveflag = False )