# kranke - January 2026
# Script to test out the class defined in epwclass.py

from   epwclass import epw_collection
import numpy as np
import matplotlib.pyplot as plt
from   physicsutils import rh_t2dpt 


tmy_collection = epw_collection( filetype = 'tmy', location = 'Boston__MA__USA' )
current_tmy    = tmy_collection.files[0]
future_tmy = current_tmy.with_futureShift( cmipdir = "/home/kranke/Documents/ResearchProjects/BC3/Data/CMIP",
                                                    params = {'model':'CanESM5', 'futyear': 2090, 'futexp':'ssp585' },
                                                    savedir = None )    
results = current_tmy.fitArma()
new_tmy = current_tmy.generatePlausible( seed = 1666 )
new_tmy2 = current_tmy.generatePlausible( seed = 1665 )


# Plot new_tmy vs current_tmy for dbt, rh, pres, wspd, and dpt
plt.figure(figsize=(12, 8))
plt.subplot(5, 1, 1)
plt.plot(current_tmy.data['dbt'], label='Original TMY', alpha=0.5)
plt.plot(new_tmy.data['dbt'], label='Generated TMY', alpha=0.5)
plt.plot(new_tmy2.data['dbt'], label='Generated TMY 2', alpha=0.5)
plt.plot(future_tmy.data['dbt'], label='Future TMY', alpha=0.5)
plt.ylabel("Temperature (°C)")
plt.subplot(5, 1, 2)
plt.plot(current_tmy.data['rh'], label='Original TMY', alpha=0.5)
plt.plot(new_tmy.data['rh'], label='Generated TMY', alpha=0.5)
plt.plot(new_tmy2.data['rh'], label='Generated TMY 2', alpha=0.5)
plt.plot(future_tmy.data['rh'], label='Future TMY', alpha=0.5)
plt.ylabel("Relative Humidity (%)")
plt.subplot(5, 1, 3)
plt.plot(current_tmy.data['pres'], label='Original TMY', alpha=0.5)
plt.plot(new_tmy.data['pres'], label='Generated TMY', alpha=0.5)
plt.plot(new_tmy2.data['pres'], label='Generated TMY 2', alpha=0.5)
plt.plot(future_tmy.data['pres'], label='Future TMY', alpha=0.5)    
plt.xlabel("Time (hours)")
plt.ylabel("Pressure (Pa)")
plt.subplot(5, 1, 4)
plt.plot(current_tmy.data['wspd'], label='Original TMY', alpha=0.5)
plt.plot(new_tmy.data['wspd'], label='Generated TMY', alpha=0.5)
plt.plot(new_tmy2.data['wspd'], label='Generated TMY 2', alpha=0.5)
plt.plot(future_tmy.data['wspd'], label='Future TMY', alpha=0.5)
plt.xlabel("Time (hours)")
plt.ylabel("Wind Speed (m/s)")
plt.subplot(5, 1, 5)
plt.plot(current_tmy.data['dpt'], label='Original TMY', alpha=0.5)
plt.plot(new_tmy.data['dpt'], label='Generated TMY', alpha=0.5)
plt.plot(new_tmy2.data['dpt'], label='Generated TMY 2', alpha=0.5)
plt.plot(future_tmy.data['dpt'], label='Future TMY', alpha=0.5)
plt.xlabel("Time (hours)")
plt.ylabel("Dew Point (°C)")
plt.tight_layout()
plt.legend()
plt.show()


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

