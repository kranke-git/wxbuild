# kranke - May 2026
# Script to run the morphing on selected locations for the paper, with all tracks

from   epwclass import epw_collection
import numpy    as np
from   plotutils import plotTrackDirect_comparison, plot_amy_famy_collections, compare_amys_ptmys


paper_locations = [ 'Singapore__Singapore', 'Helsinki__Finland', 'Cairo__Egypt', 'Washington__USA', 'Madrid__Spain' ]

# -----------------
# Input information
# -----------------
fut_year  = 2050
locations = [ 'Seattle_WA_USA' ]
ssps      = [ 'ssp126', 'ssp245', 'ssp370', 'ssp585' ]
models    = [ 'CanESM5', 'INM-CM4-8', 'MPI-ESM1-2-LR']
seeds     = np.arange( 1, 16 )

# ---------------------------------------------------------
# Track 0: Morph current tmy into the future (direct shift)
# ---------------------------------------------------------
for location in locations:
    current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
    for model in models:
        for futexp in ssps:
            print( f"Morphing location {location} with experiment {futexp} and model {model}")
            future_tmy = current_tmy_collection.with_futureShifts( params={'model': model, 'futexp': futexp, 'futyear': fut_year}, saveflag = True )


# --------------------------------------------------------------------------------------------------
# Track 1: Use the TMY to generate plausible amys in current climate and shift those into the future
# --------------------------------------------------------------------------------------------------
for location in locations:
    # Generate plausible amys in current climate from current TMY
    current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
    current_tmy            = current_tmy_collection.files[0] # Just select the first file
    for seed in seeds:
        print( f"Generating file with seed {seed}")
        plausible_amy = current_tmy.generatePlausible( seed = seed, write_flag = True )
    # Future shift
    current_ptmy_collection = epw_collection( filetype = 'ptmy', location = location )
    for model in models:
        for futexp in ssps:
            print( f"Morphing location {location} with experiment {futexp} and model {model}")
            future_ptmy = current_ptmy_collection.with_futureShifts( params={'model': model, 'futexp': futexp, 'futyear': fut_year}, saveflag = True )

# ---------------------------------------------------------
# Track 2: Morph current RMY into the future (direct shift)
# ---------------------------------------------------------
for location in locations:
    current_rmy_collection = epw_collection( filetype = 'rmy', location = location )
    for model in models:
        for futexp in ssps:
            print( f"Morphing location {location} with experiment {futexp} and model {model}")
            future_rmy = current_rmy_collection.with_futureShifts( params={'model': model, 'futexp': futexp, 'futyear': fut_year}, saveflag = True )

# -----------------------------------------------------------
# Track 3:
# (1) Morph the current AMYs (below)
# (2) Use future AMYs to generate a future RMY (this step is not done in this code)
# -----------------------------------------------------------
for location in locations:
    current_amy_collection = epw_collection( filetype = 'amy', location = location )
    for model in models:
        for futexp in ssps:
            print( f"Morphing location {location} with experiment {futexp} and model {model}")
            future_amys = current_amy_collection.with_futureShifts( params={'model': model, 'futexp': futexp, 'futyear': fut_year}, saveflag = True )

# -----------------------------------------------------------
# Plot some diagnotics for the future shifts (as quick check)
# -----------------------------------------------------------
variables = ['dbt', 'dpt', 'rh', 'pres', 'wspd', 'wdir']
location  = locations[ 0 ]
for location in locations:
    
    plotTrackDirect_comparison( location, variables, filetype='tmy' ) # Track 0
    plotTrackDirect_comparison( location, variables, filetype='rmy' ) # Track 2
    plot_amy_famy_collections( location, variables, filetype='ptmy', N = 5, models = ['CanESM5'], scenarios = ['ssp585'] ) # Track 3
    plot_amy_famy_collections( location, variables, filetype='amy', N = 5, models = ['CanESM5'], scenarios = ['ssp585'] ) # Track 3
    compare_amys_ptmys( location, variables, N = 5 )
    