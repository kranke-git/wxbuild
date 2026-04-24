# kranke - January 2026
# Script to test out the class defined in epwclass.py

from   epwclass import epw_collection
import numpy as np
import matplotlib.pyplot as plt
from   physicsutils import rh_t2dpt 


current_tmy = epw_collection( filetype = 'tmy', location = 'Boston__MA__USA' ).files[0]
future_tmy = current_tmy.with_futureShift( cmipdir = "/home/kranke/Documents/ResearchProjects/BC3/Data/CMIP",
                                                    params = {'model':'CanESM5', 'futyear': 2090, 'futexp':'ssp585' },
                                                    savedir = None )    

monthly_avg_current = current_tmy.calculateMonthlyAverages( 'rh' )
monthly_avg_future  = future_tmy.calculateMonthlyAverages( 'rh' )

plt.plot( monthly_avg_current, label='Original TMY' )
plt.plot( monthly_avg_future, label='Future TMY' )
plt.legend()
plt.title("Relative Humidity: Original vs Future TMY")
plt.xlabel("Month")
plt.ylabel("Relative Humidity (%)")
plt.grid(True, alpha=0.25)
plt.show()
# current_tmy.downloadCmip( model = 'CanESM5' )
# future_tmy  = current_tmy.files[0].with_futureShift(    cmipdir = f"{current_tmy.data_directory}/cmip6", 
#                                                         params = {'model':'CanESM5'},
#                                                         savedir = None )

# future_amy_coll = current_amy.with_futureShifts( params = { 'model':'CanESM5', 'futyear': 2050, 'futexp':'ssp585'  }, saveflag = True )

# Test on the get anomalies function
df = current_tmy.getVariableAnomalies( params = { 'variable':'dbt', 'years': np.arange( 2020, 2101 ), 'futexp':'ssp585' } )
current_tmy.data['dpt_calc'] = round( rh_t2dpt( current_tmy.data['dbt'], current_tmy.data['rh'] ), 1 )

# Test on the dbt formulation
plt.plot(current_tmy.data['dpt'], current_tmy.data['dpt_calc'], '.', alpha=0.5)
bias = ( current_tmy.data['dpt_calc'] - current_tmy.data['dpt'] ).mean()
rmse = np.sqrt( ( ( current_tmy.data['dpt_calc'] - current_tmy.data['dpt'] ) ** 2 ).mean() )
corr = np.corrcoef( current_tmy.data['dpt_calc'], current_tmy.data['dpt'] )[0,1]
print(f"Bias: {bias:.2f} °C, RMSE: {rmse:.2f} °C, Correlation: {corr:.2f}")

