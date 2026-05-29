# kranke - May 2026
# Script to define some useful plotting functions that use epwclass

import matplotlib.pyplot as plt
from   epwclass          import epw_collection
import os
import numpy             as np

def plotTrackDirect_comparison( location, variables, filetype = 'tmy', models = ['CanESM5', 'INM-CM4-8', 'MPI-ESM1-2-LR'], ssps = [ 'ssp126', 'ssp585' ] ):

    # Track 0: Morph current tmy into the future (direct shift)
    current_tmy_collection = epw_collection( filetype = filetype, location = location )
    current_tmy            = current_tmy_collection.files[0]
    future_tmy_collection  = epw_collection( filetype = f'f{filetype}', location = location )
    alpha                  = 0.5
    
    # Color map for models, linestyle map for scenarios
    model_colors = {'CanESM5': '#1f77b4', 'INM-CM4-8': '#ff7f0e', 'MPI-ESM1-2-LR': '#2ca02c'}
    ssp_linestyles = {'ssp126': '-', 'ssp245': '--', 'ssp370': '-.', 'ssp585': ':'}

    fig, axes = plt.subplots(6, 2, figsize=(14, 12))

    for i, var in enumerate(variables):
        
        # Left column: hourly data
        ax_left = axes[i, 0]
        current_data = current_tmy.data[var]
        ax_left.plot(current_data, label='Current', alpha=0.5, color='black', linewidth=2)
        
        for model in models:
            for futexp in ssps:
                # Find future file in collection
                future_tmy = [f for f in future_tmy_collection.files if model in f.filename and futexp in f.filename][0]
                future_data = future_tmy.data[ var ]
                label = f'{model} - {futexp}'
                ax_left.plot(future_data, label=label, alpha=0.6, color=model_colors[model], linestyle=ssp_linestyles[futexp])
        
        ax_left.set_title(f'{var.upper()} - Hourly Data {filetype.upper()} ')
        ax_left.set_xlabel('Hour of Year')
        ax_left.set_ylabel(var.upper())
        ax_left.grid(True, alpha=0.3)
        
        # Right column: monthly average
        ax_right = axes[i, 1]
        current_monthly = current_tmy.calculateMonthlyAverages( var )
        months = np.arange(1, 13)
        ax_right.plot(months, current_monthly, marker='o', label='Current', alpha=alpha, 
                        color='black', linewidth=2, markersize=5)
        
        for model in models:
            for futexp in ssps:
                # Load future file
                future_tmy = [f for f in future_tmy_collection.files if model in f.filename and futexp in f.filename][0]
                future_monthly = future_tmy.calculateMonthlyAverages( var )
                label = f'{model} - {futexp}'
                ax_right.plot(months, future_monthly, marker='o', label=label, alpha=alpha, color=model_colors[model], linestyle=ssp_linestyles[futexp], markersize=5)    
                
        
        ax_right.set_title(f'{var.upper()} - Monthly Average')
        ax_right.set_xlabel('Month')
        ax_right.set_ylabel(f'{var.upper()} (Monthly Avg)')
        ax_right.set_xticks(months)
        # Only show legend for the last variable to avoid clutter
        if i == len(variables) - 1:
            ax_right.legend(fontsize=8)
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Output the figure to a file
    outdir = f"{current_tmy_collection.data_directory}/{location}/diagnostics"
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)
    track_no = 0 if filetype == 'tmy' else 2 if filetype == 'rmy' else 100
    plt.savefig(f'{outdir}/track{track_no}_allVariables_comparison.png', dpi=150)
    plt.show()


def plot_amy_famy_collections( location, variables = ['dbt', 'dpt', 'rh', 'pres', 'wspd', 'wdir'], filetype = 'amy', models = ['CanESM5'], scenarios = ['ssp585'], N = None ):
    """
    Function to plot the AMY and FAMY collections for all locations.
    This is basically a proxy for track3
    """

    # Color map for models, linestyle map for scenarios
    model_colors = {'CanESM5': '#1f77b4', 'INM-CM4-8': '#ff7f0e', 'MPI-ESM1-2-LR': '#2ca02c'}
    ssp_linestyles = {'ssp126': '-', 'ssp245': '--', 'ssp370': '-.', 'ssp585': ':'}

    fig, axes = plt.subplots(6, 2, figsize=(14, 12))
    amy_collection = epw_collection( filetype = filetype, location = location )
    famy_collection = epw_collection( filetype = f"f{filetype}", location = location )
    if N is None:
        # Plot everything
        Namy  = len( amy_collection.files )
        Nfamy = len( famy_collection.files )
    else:
        Namy = N
        Nfamy = N


    for i, var in enumerate(variables):
        
        # Left column: AMY and FAMY collections on hourly data
        for amy in amy_collection.files[0:Namy]:
            axes[i, 0].plot(amy.data[ var ], label=amy.filename, alpha=0.5, color = 'gray' )
        
        for model in models:
            for scenario in scenarios:
                famys = [f for f in famy_collection.files if model in f.filename and scenario in f.filename][0:Nfamy]
                for famy in famys:
                    axes[i, 0].plot(famy.data[ var ], label=famy.filename, alpha=0.5, color = model_colors[model], linestyle = ssp_linestyles[ scenario ] )
        
        if filetype == 'ptmy':
            # add original TMY for comparison
            current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
            current_tmy = current_tmy_collection.files[0]
            axes[i, 0].plot(current_tmy.data[ var ], label=current_tmy.filename, color = 'black', linewidth = 1 )
        
        axes[i, 0].set_xlabel('Hour of Year')
        axes[i, 0].set_ylabel(f'{var.upper()} (Hourly)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Right column: AMY and FAMY collections on monthly averages
        for amy in amy_collection.files[0:Namy]:
            monthly_avg = amy.calculateMonthlyAverages( var )
            axes[i, 1].plot(np.arange(1, 13), monthly_avg, label = amy.filename, alpha=0.5, color = 'gray' )
        
        for model in models:
            for scenario in scenarios:
                famys = [f for f in famy_collection.files if model in f.filename and scenario in f.filename][0:Nfamy]
                for famy in famys:
                    monthly_avg = famy.calculateMonthlyAverages( var )
                    axes[i, 1].plot(np.arange(1, 13), monthly_avg, label=famy.filename, alpha=0.5, color=model_colors[model], linestyle=ssp_linestyles[scenario])
        
        if filetype == 'ptmy':
            # add original TMY for comparison
            current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
            current_tmy = current_tmy_collection.files[0]
            monthly_avg = current_tmy.calculateMonthlyAverages( var )
            axes[i, 1].plot(np.arange(1, 13), monthly_avg, label=current_tmy.filename, alpha=1, color='black', linewidth=1)
        
        axes[i, 1].set_xlabel('Month')
        axes[i, 1].set_ylabel(f'{var.upper()} (Monthly Avg)')
        axes[i, 1].set_xticks(np.arange(1, 13))
        axes[i, 1].grid(True, alpha=0.3)

    # Final touches
    plt.tight_layout()
    # Output the figure to a file
    outdir = f"{amy_collection.data_directory}/{location}/diagnostics"
    if os.path.exists( outdir ) == False:
        os.makedirs( outdir )
    track_no = 3 if filetype == 'amy' else 1 if filetype == 'ptmy' else 100
    plt.savefig(f'{outdir}/track{track_no}_allVariables_comparison.png', dpi=150)
    plt.show()
    
def compare_amys_ptmys( location, variables, N = 5 ):
    
    """
    Function to compare the AMY and PTMY collections for a given location.
    """
    
    fig, axes       = plt.subplots(6, 2, figsize=(14, 12))
    amy_collection  = epw_collection( filetype = 'amy',  location = location )
    ptmy_collection = epw_collection( filetype = 'ptmy', location = location )
    
    if N is None:
        # Plot everything
        Namy  = len( amy_collection.files )
        Nptmy = len( ptmy_collection.files )
    else:
        Namy = N
        Nptmy = N

    for i, var in enumerate( variables ):
        
        # Left column: AMY and PTMY collections on hourly data
        for amy in amy_collection.files[0:Namy]:
            axes[i, 0].plot(amy.data[ var ], label=amy.filename, alpha=0.5, color = 'gray' )
        for ptmy in ptmy_collection.files[0:Nptmy]:
            axes[i, 0].plot(ptmy.data[ var ], label=ptmy.filename, alpha=0.5, color = 'lightblue' )

            # add original TMY for comparison
            current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
            current_tmy = current_tmy_collection.files[0]
            axes[i, 0].plot(current_tmy.data[ var ], label=current_tmy.filename, color = 'black', linewidth = 1 )
        
        axes[i, 0].set_xlabel('Hour of Year')
        axes[i, 0].set_ylabel(f'{var.upper()} (Hourly)')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Right column: AMY and PTMY collections on monthly averages
        for amy in amy_collection.files[0:Namy]:
            monthly_avg = amy.calculateMonthlyAverages( var )
            axes[i, 1].plot(np.arange(1, 13), monthly_avg, label = amy.filename, alpha=0.5, color = 'gray' )
        for ptmy in ptmy_collection.files[0:Nptmy]:
            monthly_avg = ptmy.calculateMonthlyAverages( var )
            axes[i, 1].plot(np.arange(1, 13), monthly_avg, label = ptmy.filename, alpha=0.5, color = 'lightblue' )
                
        # add original TMY for comparison
        current_tmy_collection = epw_collection( filetype = 'tmy', location = location )
        current_tmy = current_tmy_collection.files[0]
        monthly_avg = current_tmy.calculateMonthlyAverages( var )
        axes[i, 1].plot(np.arange(1, 13), monthly_avg, label=current_tmy.filename, alpha=1, color='black', linewidth=1)
        
        axes[i, 1].set_xlabel('Month')
        axes[i, 1].set_ylabel(f'{var.upper()} (Monthly Avg)')
        axes[i, 1].set_xticks(np.arange(1, 13))
        axes[i, 1].grid(True, alpha=0.3)

    # Final touches
    plt.tight_layout()
    # Output the figure to a file
    outdir = f"{amy_collection.data_directory}/{location}/diagnostics"
    if os.path.exists( outdir ) == False:
        os.makedirs( outdir )
    plt.savefig(f'{outdir}/amys_vs_ptmys_allVariables_comparison.png', dpi=150)
    plt.show()

