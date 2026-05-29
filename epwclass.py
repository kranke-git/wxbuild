# kranke - January 2026
# Script to define the main epwdata class for handling EPW files

import re
import os
import requests
import pandas       as     pd
import numpy        as     np
import copy
import warnings
import xarray       as     xr
from   ioutils      import list_svante_files, read_nth_line
from   miscutils    import shift_tuple, swapMonthTmy
from   constants    import epw_colnames, months_labels
from   dataclasses  import dataclass, replace
from   cmip6utils   import CalcGlobalDT, getPatternCoefficients, calculateShift
from   pathlib      import Path
import statsmodels.api as sm
from   armaUtils     import fitDiurnalCycle, checkResiduals, RegressArma
from   physicsutils import rh_t2dpt


@dataclass
class EPWFile:
    
    file_path:       str
    filetype:        str
    location:        str
    filename:        str          = None
    long_name:       str          = None
    state:           str          = None
    country:         str          = None
    source:          str          = None
    statid:          str          = None
    data:            pd.DataFrame = None
    years_in_file:   list         = None
    qc:              bool         = True
    latitude:        float        = None
    longitude:       float        = None
    year_range:      tuple        = None
    avgYear:         int          = None
    timezone:        float        = None
    elevation:       float        = None
    design_string:   str          = None
    extreme_string:  str          = None
    ground_string:   str          = None
    daylight_string: str          = None
    comment1:        str          = None
    comment2:        str          = None
    data_period_str: str          = None
    arma_values:     dict         = None
    seed:          int            = None
    _skip_post:      bool         = False 
    
    def __post_init__( self ):
        """
            Do five things after initialization:
            (1) Read the EPW file DATA into a DataFrame, assign filename field based on path
            (2) Automatically populate years from the DataFrame
            (3) Quality Check the data as follows:
            ---- Check if specified columns in a DataFrame have only one unique value.
            ---- Also check for absurd values in dbt, dpt, rh
            (4) Get range of years if available
            (5) Assign other metadata fields from the header
        """
        # Skip post init if specified   
        if self._skip_post is True:
            return
        
        # ( 1 ) Read the EPW file into a DataFrame, assign filename field based on path 
        self.filename = Path( self.file_path ).name
        if self.data is None:
            # In case no data is provided, read from file (data could be supplied by methods like futureShift)
            print(f"Reading file: { self.file_path }")
            df              = pd.read_csv( self.file_path, skiprows = 8, header = None, names = epw_colnames, index_col = False )
            df['datetime']  = pd.to_datetime({ 'year': df['Year'], 'month': df['Month'], 'day': df['Day'], 'hour': df['Hour']-1 }, errors = 'coerce' )
            df['date']      = df['datetime'].dt.date
            self.data       = df
        else:
            df = self.data
        # Assign na if any value is 999
        df.replace( 999, np.nan, inplace = True )

        # ( 2 ) Automatically populate years from the DataFrame
        if isinstance( self.data, pd.DataFrame)  and 'Year' in self.data.columns:
            self.years_in_file = self.data['Year'].unique().tolist()
        # ( 3 ) Do QC on initialization
        checkColumns      = ['dbt', 'dpt', 'rh', 'wspd', 'wdir' ]   
        uniqueValuesCheck = { col: df[col].nunique() == 1 for col in checkColumns }
        dbt_out_of_range  = ( (df[ 'dbt'] < -100) | ( df[ 'dbt' ] > 200 )).any()
        dpt_out_of_range  = ( (df[ 'dpt'] < -100) | ( df[ 'dpt' ] > 200 )).any()
        rh_out_of_range   = ( (df[ 'rh'] < -50) | ( df[ 'rh' ] > 150 )).any()
        checks = [  ("constant_value", any(uniqueValuesCheck.values())),
                    ("dbt_out_of_range", dbt_out_of_range),
                    ("dpt_out_of_range", dpt_out_of_range),
                    ("rh_out_of_range", rh_out_of_range) ]
        # Print failed checks
        for name, failed in checks:
            if failed:
                print( f'--- {self.filename} failed {name} QC')
                self.qc = False
            else:
                self.qc = True
        if os.path.exists( self.file_path ):
            # ( 4 ) get range of years if available
            line6 = read_nth_line( self.file_path, 6 )    
            match = re.search(r"Period of Record\s*=\s*(\d{4})-(\d{4})", line6 )
            if match:
                start_year, end_year = match.groups()
                self.year_range = ( int(start_year), int(end_year) )
                self.avgYear    = round( ( int(start_year) + int(end_year) ) / 2 )
            else:
                self.year_range = ( min( self.years_in_file ), max( self.years_in_file ) )
                self.avgYear    = round( sum( self.years_in_file ) / len( self.years_in_file ) )
            # ( 5 ) Assign other metadata fields from the header
            first_line           = read_nth_line( self.file_path, 1 )
            parts                = first_line.split( ',' )
            self.long_name       = parts[1].strip() if len(parts) > 1           else None
            self.state           = parts[2].strip() if len(parts) > 2           else None
            self.country         = parts[3].strip() if len(parts) > 3           else None
            self.source          = parts[4].strip() if len(parts) > 4           else None
            self.statid          = parts[5].strip() if len(parts) > 5           else None
            self.latitude        = float( parts[6].strip() ) if len(parts) > 6  else None
            self.longitude       = float( parts[7].strip() ) if len(parts) > 7  else None
            self.timezone        = float( parts[8].strip() ) if len(parts) > 8  else None
            self.elevation       = float( parts[9].strip() ) if len(parts) > 9  else None 
            self.design_string   = read_nth_line( self.file_path, 2 )
            self.extreme_string  = read_nth_line( self.file_path, 3 )
            self.ground_string   = read_nth_line( self.file_path, 4 )
            self.daylight_string = read_nth_line( self.file_path, 5 )
            self.comment1        = read_nth_line( self.file_path, 6 )
            self.comment2        = read_nth_line( self.file_path, 7 )
            self.data_period_str = read_nth_line( self.file_path, 8 )
            if self.filetype == 'ptmy' or self.filetype == 'fptmy':
                self.seed = int( re.search( r"seed(\d+)", self.filename ).group(1) ) if re.search( r"seed(\d+)", self.filename ) else None

        
    def __repr__( self ):

        nrows          = len( self.data )
        fitted = self.arma_values is not None
        return (
            f"EPWFile("
            f"nrows={nrows}, "
            f"file_path='{self.file_path}', "
            f"filetype='{self.filetype}', "
            f"location='{self.location}', "
            f"latitude={self.latitude}, "
            f"longitude={self.longitude}"
            f")"
        )
        
    def with_futureShift( self, cmipdir, params, savedir = None, verbose = False ):
        """
        Method to generate a future file from the available present-day files.
        This is non-destructive; it returns a new EPWFile instance with the future data.
        Parameters
        ----------
        """
        
        # Unpack the parameters
        model       = params.get( 'model',  'MPI-ESM1-2-LR' )
        member      = params.get( 'member', 'MAVG' )
        futyear     = params.get( 'futyear', 2050 )
        futexp      = params.get( 'futexp', 'ssp245' )
        pattern_exp = params.get( 'pattern_exp', 'ssp126-ssp245-ssp370-ssp585' )
        grid        = params.get( 'grid', 'r180x90' )
        
        # New attributes for future file
        seed_value        = self.seed if hasattr( self, 'seed' ) else None
        seed_string       = f"_seed{seed_value}" if seed_value is not None else ""
        new_filetype      = f"f{self.filetype}"
        new_filename      = f"{self.location}_{new_filetype}_{futyear}_{futexp}_{model}{seed_string}.epw"
        
        # Set the new filepath
        if savedir is not None:
            new_filepath = os.path.join( savedir, new_filename )
        else:
            new_filepath = os.path.join( os.path.dirname( self.file_path ), new_filename )

        # If file already exists, return the existing file instead of creating a new one
        if os.path.exists( new_filepath ):
            print( f"Future file {new_filepath} already exists. Returning existing file." )
            return EPWFile( file_path = new_filepath, filetype = new_filetype, location = self.location, _skip_post = False )
        
        # We only reach here if file does not exist, so we proceed to create a new future file
        # Instantiate a new EPWFile object for the future data
        new_data = self.data.copy()
        
        model_dir = f"{cmipdir}/{model}"
        # Figure out historical period from the file
        histperiod = self.year_range
        # Calculate global DT with the specified future year and the years in file
        futperiod  = shift_tuple( self.year_range, futyear )
        deltaTG    = CalcGlobalDT( model_dir, model, member, histperiod, futperiod, futexp )
        for month in np.arange( 0, 12, 1 ) + 1:
            # Figure out the average shift for the futuremonth
            idxmonth     = self.data.index[ self.data['Month'] == month].tolist() 
            coefs        = getPatternCoefficients( model_dir, pattern_exp, member, grid, month, {'lat':self.latitude, 'lon':self.longitude + 360 }, verbose = verbose )
            currentPres  = self.data[ self.data['Month'] == month]['pres'].mean()
            currentDpt   = self.data[ self.data['Month'] == month]['dpt'].mean()
            avgShift     = calculateShift( coefs, deltaTG, self.data.iloc[ idxmonth ] )
            new_data     = swapMonthTmy( new_data, idxmonth, avgShift, swapYears = np.arange( futperiod[0], futperiod[1] + 1, 1 ) )

        # Assign new headers for future file
        month_year_pairs = new_data[['Month', 'Year']].drop_duplicates().sort_values(['Month','Year'])
        num_years        = futperiod[-1] - futperiod[0] + 1
        month_strs       = [f"{months_labels[row.Month-1]}={row.Year}" for row in month_year_pairs.itertuples(index=False)]
        new_comment1     = f'COMMENTS 1,"BC3 emulator - #years=[{num_years}] Period of Record={futperiod[0]}-{futperiod[-1]}; ' + "; ".join(month_strs) + '"'
        new_comment2     = f'COMMENTS 2,"{new_filetype.upper()} processed with BC3 Emulator -- pgiani@mit.edu for more info; model={model}; scenario={futexp}"'
        new_source       = f'BC3Emulator_{model}_{member}_{futexp}_{futyear}'

        # Return a new EPWFile instance with the modified data (Replace skips the post_init method)
        new_years_in_file = new_data['Year'].unique().tolist()
        new_years_range   = ( futperiod[0], futperiod[ -1 ] )
        new_avgYear       = round( ( futperiod[ 0 ] + futperiod[ -1 ] ) / 2 )
        new_instance     = replace( self, data = new_data, file_path = new_filepath, filetype = new_filetype, filename = new_filename, years_in_file = new_years_in_file,
                        year_range = new_years_range, avgYear = new_avgYear, comment1 = new_comment1, comment2 = new_comment2, source = new_source, _skip_post = True ) 
        
        # Write out the file if savedir is specified
        if savedir is not None:
            new_instance.writeToFile( new_filepath )
            
        # Return the replaced instance
        return new_instance            

            
    def writeToFile( self, output_path = None ):
        """
        Method to write the EPWFile data to a specified output path.
        Parameters
        ----------
        output_path : str
            The file path where the EPW data should be written.
        """
        if output_path is None:
            output_path = self.file_path
        # Extract directory from output_path and create it if it doesn't exist
        output_dir = os.path.dirname( output_path )
        if not os.path.exists( output_dir ):
            os.makedirs( output_dir )
        # First write the actual data without headers, then add the headers afterwards
        # Open the output file and prepend the first 8 lines
        with open( output_path, 'w' ) as f:
            # Header lines
            f.write( f"{self.location},{self.long_name},{self.state},{self.country},{self.source},{self.statid},{self.latitude},{self.longitude},{self.timezone},{self.elevation}\n" )
            f.write( f"{self.design_string}\n" )
            f.write( f"{self.extreme_string}\n" )
            f.write( f"{self.ground_string}\n" )
            f.write( f"{self.daylight_string}\n" )
            f.write( f"{self.comment1}\n" )
            f.write( f"{self.comment2}\n" )
            f.write( f"{self.data_period_str}\n" )
            # Data
            data_to_write = self.data.drop( columns=['date', 'datetime'] )
            # Put 999 when na is present
            data_to_write.fillna( 999, inplace = True )
            data_to_write.to_csv( f, index = False, header = False )
            
    def calculateMonthlyAverages( self, variable: str ):
        """
        Method to calculate monthly averages for a specified variable.
        Parameters
        ----------
        variable : str
            The variable for which to calculate monthly averages (e.g., 'dbt', 'rh', etc.)
        Returns
        -------
        pd.Series
            A Series containing the monthly averages for the specified variable.
        """
        if variable not in self.data.columns:
            raise ValueError( f"Variable '{variable}' not found in data columns." )
        monthly_averages = self.data.groupby( 'Month' )[ variable ].mean()
        return monthly_averages

    def fitArma( self, dbt_arma_order = (2,0,2) ):

        if self.arma_values is not None:
            # Models are already fitted, return existing results
            print("ARMA models already fitted. Returning existing results.")
            return self.arma_values
        
        else:
            # Arma values not present yet, so fit the models
            print("Learning the ARMA models...")
            der_variables = ['rh', 'pres', 'wspd']
            results = {}

            for month in range(1, 13):

                month_subset = self.data[self.data['Month'] == month]
                results[month] = {}

                # DBT
                with warnings.catch_warnings():
        
                    warnings.filterwarnings(
                        "ignore",
                        message="Non-stationary starting autoregressive parameters"
                    )

                    warnings.filterwarnings(
                        "ignore",
                        message="Non-invertible starting MA parameters"
                    )
                    
                    warnings.filterwarnings(
                        "ignore",
                        message="Maximum Likelihood optimization failed to converge"
                    )
                    
                    dbtCycle, A, phi, c = fitDiurnalCycle( month_subset, month, 'dbt' )
                    armaDbt    = sm.tsa.ARIMA( month_subset['dbt'].to_numpy() - dbtCycle.to_numpy(),  order = dbt_arma_order, trend='n' ).fit()
                    dbtRes     = armaDbt.resid
                    ldbt, sdbt = checkResiduals( dbtRes )

                results[month]['dbt'] = {
                    'arma_model': armaDbt,
                    'residuals': dbtRes,
                    'residual_stats': {
                        'l': ldbt,
                        's': sdbt,
                    },
                    'diurnal_cycle': {
                        'dbtCycle': dbtCycle,
                        'A': A,
                        'phi': phi,
                        'c': c,
                    }
                }

                # Derived variables
                for var in der_variables:

                    if var == 'rh':
                        exog = month_subset['dbt']
                        armaOrd = (3, 0, 0)

                    elif var == 'pres':
                        exog = month_subset['dbt']
                        armaOrd = (5, 0, 0)

                    elif var == 'wspd':
                        exog = month_subset['dbt']
                        armaOrd = (3, 0, 0)

                    corrVar, slpVar, intVar, modelVar, armaVar, varRes = ( RegressArma( month_subset[var], exog=exog, armaOrd=armaOrd ) )
                    lVar, sVar = checkResiduals(varRes)

                    results[month][var] = {
                        'arma_model': armaVar,
                        'residuals': varRes,
                        'residual_stats': {
                            'l': lVar,
                            's': sVar,
                        },
                        'regression': {
                            'corr': corrVar,
                            'slope': slpVar,
                            'intercept': intVar,
                            'model': modelVar,
                        }
                    }
            
            # Assign the results to an attribute for later use
            self.arma_values = results
            return results
    
    def generatePlausible( self, seed = 1666, write_flag = False ):
        """
        Method to generate a plausible future EPWFile based on the learned ARMA models.
        This method uses the fitted ARMA models to simulate future weather data.
        seed: int, optional
            Random seed for reproducibility (default: 1666).
        Returns
        -------
        EPWFile
            A new EPWFile instance containing the simulated future data.
        """
        
        # Get new filename and file path for the plausible future file
        p            = Path(self.filename)
        new_filename = f"{p.stem}_plausible_seed{seed}{p.suffix}"
        new_filetype = f"p{self.filetype}"
        new_filepath = os.path.join( os.path.dirname( self.file_path ), new_filename ).replace( self.filetype, new_filetype )
        
        # Check if file already exists; if so, return the existing file instead of generating a new one
        if os.path.exists( new_filepath ):
            print( f"Plausible file {new_filepath} with seed {seed} already exists. Returning existing file." )
            return EPWFile( file_path = new_filepath, filetype = new_filetype, location = self.location, _skip_post = False )
        
        # Check if ARMA models are fitted; if not, fit them
        if self.arma_values is None:
            print("ARMA models not fitted yet. Fitting now...")
            self.fitArma()
        else:
            print("Using existing fitted ARMA models.")
        
        # Loop over months and generate new data based on the learned ARMA models
        rng     = np.random.default_rng( seed )
        new_epw = copy.deepcopy( self ) # Create a deep copy of the current EPWFile instance to hold the new data
        
        roundings = { 'dbt': 1, 'rh': 0, 'pres': -1, 'wspd': 1 }
        # Loop every month
        for month in range(1, 13):
            
            month_mask = new_epw.data['Month'] == month
            dbtRes     = self.arma_values[month]['dbt']['residuals']
            armaDbt    = self.arma_values[month]['dbt']['arma_model']
            dbtCycle   = self.arma_values[month]['dbt']['diurnal_cycle']['dbtCycle']
            
            # for dbt, the generation is dbtCycle (already correct length) + residuals simulated from the ARMA model (resampling from the residuals as the empirical distribution)
            dbtGen  = round( dbtCycle + armaDbt.simulate( nsimulations = len( dbtRes ), state_shocks = rng.choice( dbtRes, size = len( dbtRes ), replace = True ), random_state = rng ), roundings[ 'dbt' ] ) 
            new_epw.data.loc[ month_mask, 'dbt' ] = dbtGen.values
            # for all the other variables, regression upon dbt + ARMA residuals
            for var in ['rh', 'pres', 'wspd']:
                varRes   = self.arma_values[month][var]['residuals']
                armaVar  = self.arma_values[month][var]['arma_model']
                modelVar = self.arma_values[month][var]['regression']['model']
                # Generate the new variable based on the regression model and the simulated residuals
                varGen   = round( modelVar.predict( dbtGen.values.reshape(-1,1) ) + armaVar.simulate( nsimulations = len( varRes ), state_shocks = rng.choice( varRes, size = len( varRes ), replace = True ), random_state = rng ), roundings[ var ] )
                new_epw.data.loc[ month_mask, var ] = varGen.values
                
            # Fix RH to be within min( existing_rh ) and 100%, and also recalculate dewpoint based on the new dbt and rh
            min_rh = round( self.data.loc[ month_mask, 'rh' ].min() * 0.90 ) # Allow for a 10% decrease in minimum RH to avoid unrealistic low values
            max_rh = 100 
            new_epw.data.loc[ month_mask, 'rh' ]  = new_epw.data.loc[ month_mask, 'rh' ].clip( lower = min_rh, upper = max_rh )
            new_epw.data.loc[ month_mask, 'dpt' ] = round( rh_t2dpt( new_epw.data.loc[ month_mask, 'dbt' ], new_epw.data.loc[ month_mask, 'rh' ] ), 1 )
            
            # Fix pressure to be within min( existing_pressure ) and max( existing_pressure )
            min_pres = round( self.data.loc[ month_mask, 'pres' ].min() * 0.95 ) # Allow for a 5% decrease in minimum pressure to avoid unrealistic low values
            max_pres = round( self.data.loc[ month_mask, 'pres' ].max() * 1.05 ) # Allow for a 5% increase in maximum pressure to avoid unrealistic high values
            new_epw.data.loc[ month_mask, 'pres' ] = new_epw.data.loc[ month_mask, 'pres' ].clip( lower = min_pres, upper = max_pres )
            # Clip wind speed values to be non-negative and bounded by the maximum observed wind speed in the original data for that month
            max_wind = round( self.data.loc[ month_mask, 'wspd' ].max() * 1.10 ) # Allow for a 10% increase in maximum wind speed to avoid unrealistic high values
            min_wind = 0
            new_epw.data.loc[ month_mask, 'wspd' ] = new_epw.data.loc[ month_mask, 'wspd' ].clip( lower = min_wind, upper = max_wind )
        
        # Change some attributes in the new_epw instance to reflect that it is a generated plausible future file
        new_epw.file_path = new_filepath
        new_epw.filetype  = new_filetype
        new_epw.filename  = new_filename
        new_epw.comment2  = f'COMMENTS 2," PLAUSIBLE (seed={seed}) {new_epw.filetype.upper()} file generated with BC3 Emulator (i.e., not real measurements) -- pgiani@mit.edu for more info"'
        new_epw.seed      = seed
        # Write if requested
        if write_flag is True:
            new_epw.writeToFile()
        
        return new_epw

class epw_collection:
    def __init__(self, filetype: str, location: str, data_directory: str = "./epwdata", search_online: bool = True ):
        """
        Parameters
        ----------
        filetype : str
            Type of file ('tmy', 'amy', 'rmy', 'ftmy', 'famy', 'frmy')
        location : str
            File location or identifier
        data_directory : str, optional
            Base directory for data files (default: './epwdata/')
        search_online : bool, optional
            Whether to search online for files (default: True)
        """
        
        self.obj_type         = filetype
        self.location         = location
        self.data_directory   = data_directory   
        self.online_directory = 'https://svante.mit.edu/~pgiani/wxbuild_data'  
        # Make sure local directory exists
        os.makedirs( f"{self.data_directory}/{self.location}/{self.obj_type}", exist_ok = True )
        # Search files from local directory first; look online if not found locally and search_online is True
        self.files = os.listdir( f"{self.data_directory}/{self.location}/{self.obj_type}" )
        
        if self.files == []:
            print( f"No local files found for {self.location}/{self.obj_type}.")
            if search_online is True:
                print( f"Searching svante directory for files..." )            
                self.files = list_svante_files(f"{self.online_directory}/{self.location}/{self.obj_type}")
                if self.files == 404:
                    print( f"No online files found for {self.location}/{self.obj_type}.")
                    self.files = []
                else:
                    print(f"Found {len(self.files)} {self.obj_type} files online for {self.location}.")
                    # Copy them locally
                    for file in self.files:
                        file_url  = f"{self.online_directory}/{self.location}/{self.obj_type}/{file}"
                        local_path= f"{self.data_directory}/{self.location}/{self.obj_type}/{file}"
                        print(f"--- Downloading {file_url} to {local_path}")
                        resp      = requests.get( file_url )
                        resp.raise_for_status()
                        with open( local_path, "wb" ) as f:
                            f.write( resp.content )
                            
        # Count the files after both operations; if none found even online, raise error
        self.Nfiles   = len( self.files )
        if self.files == []:
            raise ValueError(f"No {self.obj_type} files found in the specified directory.")
        else:
            self.files = self.read_all_files()
        # Set amy_years if filetype is 'amy'
        if self.obj_type == 'amy':
            self.amy_years = [ file.avgYear for file in self.files ]
    
    def __repr__( self ):
        return (
            f"epw_collection("
            f"obj_type='{self.obj_type}', "
            f"location='{self.location}', "
            f"Nfiles={self.Nfiles}, "
            f"amy_years={getattr(self, 'amy_years', None)}"
            f")"
        )

    
    def read_all_files( self ):
        """
        Method to read all EPW files in the specified directory and store them as EPWFile instances.
        If files are read from a web directory, also copy them locally.
        Returns a list of EPWFile instances.
        """
        epw_files = []
        for file in self.files:
            file_path        = os.path.join( self.data_directory, self.location, self.obj_type, file )
            epw_file         = EPWFile( file_path = file_path, filetype = self.obj_type, location = self.location )
            if epw_file.qc is True:
                epw_files.append( epw_file )
            else:
                print( f"File {file} failed quality checks and will be skipped." )
        return epw_files

    def downloadCmip( self, model: str ):
        # Create local directory if it doesn't exist
        # Download the files
        print( f"Downloading CMIP6 files for {model}... It might take a few minutes." )
        cmip6_files = list_svante_files( f"{self.online_directory}/cmip6/{model}", extension = ".nc" )
        if cmip6_files == 404:
            raise ValueError( f"CMIP6 files for model {model} not found online." )
        os.makedirs( f"{self.data_directory}/cmip6/{model}", exist_ok = True )
        for file in cmip6_files:
            file_url   = f"{self.online_directory}/cmip6/{model}/{file}"
            local_path = f"{self.data_directory}/cmip6/{model}/{file}"
            if not os.path.exists( local_path ):
                resp = requests.get( file_url )
                resp.raise_for_status()
                with open( local_path, "wb" ) as f:
                    f.write( resp.content )

    def with_futureShifts( self, params: dict, saveflag: bool = False ):
        """
        Method to generate future shifted files for all EPWFile instances in the collection.
        Parameters
        ----------
        params : dict
            Dictionary containing parameters for future shift (model, member, futyear, futexp, pattern_exp, grid)
        saveFlag : bool, optional
            Whether to save the modified files (default: False)
        """
        
        # Set output directory if saving is requested
        if saveflag is True:
            savedir = f"{self.data_directory}/{self.location}/f{self.obj_type}"
            os.makedirs( savedir, exist_ok = True )
        else:
            savedir = None
            
        # Set the default model if not provided
        if 'model' not in params:
            params['model'] = 'MPI-ESM1-2-LR'
            
        # Download CMIP6 files if not already present
        model_dir = f"{self.data_directory}/cmip6/{ params['model'] }"
        if os.path.exists( model_dir ) is False:
            self.downloadCmip( params['model'] )
        else:
            print( f"CMIP6 files for {params['model']} already exist locally. Proceeding with future shift..." )
            
        # Calculate yearsShift for 'amy' files
        if self.obj_type == 'amy':
            yearsShift    = params.get( 'futyear', 2050 ) - round( sum( self.amy_years ) / len( self.amy_years ) )
            print(f"Calculated yearsShift = {yearsShift} based on average year of current AMY files and futyear parameter.")
            # Check that the yearsShift would not bring any of the current amy_years beyond 2100; if so, adjust
            max_future_year = max( self.amy_years ) + yearsShift
            if max_future_year > 2100:
                yearsShift = 2100 - max( self.amy_years )
                print( f"Adjusted yearsShift t{yearsShift} to avoid exceeding year 2100." )
        
        # Loop over all files and generate future shifted versions
        future_files  = []
        for epwfile in self.files:
            if self.obj_type == 'amy':
                params['futyear'] = epwfile.avgYear + yearsShift
            else:
                params['futyear'] = params.get( 'futyear', 2050 )
            future_files.append( epwfile.with_futureShift( f"{self.data_directory}/cmip6", params, savedir = savedir ) )
        
        # Set attributes for the new collection
        self_copy           = copy.deepcopy( self )
        self_copy.files     = future_files
        self_copy.obj_type  = f"f{self.obj_type}"
        self_copy.amy_years = [ file.avgYear for file in future_files ] if self.obj_type == 'amy' else None
        return self_copy

    def getVariableAnomalies( self, params ):
        """
            Function to calculate the anomalies for a given variable and month
            Args:
                location (str): Location for which to calculate the anomalies
                variable (str): Variable for which to calculate the anomalies
                years (list):    List of years to consider for the anomalies
            Returns:
                pd.Series: Series containing the anomalies for the given variable
        """
        varMapping  = { 'dbt': 'tas'}
        variable    = params.get( 'variable', 'dbt' )
        years       = params.get( 'years', [ 2050 ] )
        model       = params.get( 'model', 'MPI-ESM1-2-LR' )
        cmipdir     = params.get( 'cmipdir', f"./epwdata/cmip6/{model}" )
        member      = params.get( 'member', 'MAVG' )
        futexp      = params.get( 'futexp', 'ssp245' )
        experiments = "ssp126-ssp245-ssp370-ssp585"
        grid        = "r180x90"
        if variable not in varMapping:
            raise ValueError( f"Variable {variable} not recognized. Available variables: {list(varMapping.keys())}" )
        
        # Get Annual averages for the variable from the file
        if variable not in self.files[0].data.columns:
            raise ValueError( f"Variable {variable} not found in the EPW file data columns." )
        else:
            avgVar = self.files[0].data[ variable ].mean()
        
        # Get CMIP6 files if not already present
        if os.path.exists( cmipdir ) is False:
            self.downloadCmip( model )
        
        # Get CMIP6 pattern scaling coefficients for the location and variable
        var_cmip = varMapping[ variable ]
        n4file  = f"{cmipdir}/PatternScalingCoefficients_{var_cmip}_{experiments}_{member}_{grid}_AnnualAverages.nc"
        if os.path.isfile( n4file ):
            coefs    = xr.open_dataset( n4file )['slope'].sel(**{'lat':self.files[0].latitude, 'lon':self.files[0].longitude + 360 }, method = 'nearest' ).values
        else:  
            print( f"Downloading pattern scaling coefficients for {var_cmip} from svante directory..." )
            n4file_url = f"{self.online_directory}/cmip6/{model}/PatternScalingCoefficients_{var_cmip}_{experiments}_{member}_{grid}_AnnualAverages.nc"
            resp       = requests.get( n4file_url )
            resp.raise_for_status()
            with open( n4file, "wb" ) as f:
                f.write( resp.content )
            coefs    = xr.open_dataset( n4file )['slope'].sel(**{'lat':self.files[0].latitude, 'lon':self.files[0].longitude + 360 }, method = 'nearest' ).values
        # Loop over the years
        print( f"Computing anomalies..." )
        var_anomalies = []
        for year in years:
            dt_global = CalcGlobalDT( cmipdir, model, member, self.files[0].year_range, (year, year), futexp )
            var_anomaly = coefs * dt_global
            var_anomalies.append( var_anomaly )
            
        dfreturn = pd.DataFrame(
            {'year': years, f'{variable}_anomaly': var_anomalies}
        ).set_index('year')    
        dfreturn[ variable ] = dfreturn[f'{variable}_anomaly'] + avgVar
        return dfreturn
