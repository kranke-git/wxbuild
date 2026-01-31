# kranke - January 2026
# Script to test out the class defined in epwclass.py

from epwclass import epw_collection

current_tmy = epw_collection( filetype = 'tmy', location = 'Boston__MA__USA' )
# current_tmy.downloadCmip( model = 'CanESM5' )
# future_tmy  = current_tmy.files[0].with_futureShift(    cmipdir = f"{current_tmy.data_directory}/cmip6", 
#                                                         params = {'model':'CanESM5'},
#                                                         savedir = None )

future_tmy_coll = current_tmy.with_futureShifts( params = { 'model':'CanESM5' }, saveflag = False )