# kranke - January 2026
# Script to define the main epwdata class for handling EPW files

import re
import os
import requests
import pandas       as     pd
import numpy        as     np
from   ioutils      import list_svante_files, read_nth_line
from   miscutils    import shift_tuple, swapMonthTmy
from   constants    import epw_colnames
from   dataclasses  import dataclass, field, replace
from   cmip6utils   import CalcGlobalDT, getPatternCoefficients, calculateShift
from   pathlib      import Path

@dataclass
class EPWFile:
    
    file_path:     str
    filetype:      str
    location:      str
    filename:      str          = None
    data:          pd.DataFrame = None
    years_in_file: list         = None
    qc:            bool         = True
    latitude:      float        = None
    longitude:     float        = None
    year_range:    tuple        = None
    avgYear:       int          = None
    
    def __post_init__( self ):
        """
            Do five things after initialization:
            (1) Read the EPW file into a DataFrame, assign filename field based on path
            (2) Automatically populate years from the DataFrame
            (3) Quality Check the data as follows:
            ---- Check if specified columns in a DataFrame have only one unique value.
            ---- Also check for absurd values in dbt, dpt, rh
            (4) Get latitude and longitude
            (5) Get range of years if available
        """
        
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
            # ( 4 ) Get latitude and longitude from filename if possible
            dat = pd.read_csv( self.file_path, nrows = 1, header = None )
            lat = dat.iloc[ 0, 6 ]
            lon = dat.iloc[ 0, 7 ]
            if lon < 0:
                lon = 360 + lon
            self.latitude  = lat
            self.longitude = lon
            # ( 5 ) get range of years if available
            line6 = read_nth_line( self.file_path, 6 )    
            match = re.search(r"Period of Record\s*=\s*(\d{4})-(\d{4})", line6 )
            if match:
                start_year, end_year = match.groups()
                self.year_range = ( int(start_year), int(end_year) )
                self.avgYear    = round( ( int(start_year) + int(end_year) ) / 2 )
            else:
                self.year_range = ( min( self.years_in_file ), max( self.years_in_file ) )
                self.avgYear    = round( sum( self.years_in_file ) / len( self.years_in_file ) )
        
    def with_futureShift( self, cmipdir, params, savedir = None ):
        """
        Method to generate a future file from the available present-day files.
        This is non-destructive; it returns a new EPWFile instance with the future data.
        Parameters
        ----------
        """
        # Unpack the parameters
        model       = params.get( 'model',  'CanESM5' )
        member      = params.get( 'member', 'MAVG' )
        futyear     = params.get( 'futyear', 2050 )
        futexp      = params.get( 'futexp', 'ssp245' )
        pattern_exp = params.get( 'pattern_exp', 'ssp126-ssp245-ssp370-ssp585' )
        grid        = params.get( 'grid', 'r180x90' )
        
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
            coefs        = getPatternCoefficients( model_dir, pattern_exp, member, grid, month, {'lat':self.latitude, 'lon':self.longitude } )
            currentPres  = self.data[ self.data['Month'] == month]['pres'].mean()
            currentDpt   = self.data[ self.data['Month'] == month]['dpt'].mean()
            avgShift     = calculateShift( coefs, deltaTG, currentPres, currentDpt, self.data.iloc[ idxmonth ] )
            new_data     = swapMonthTmy( new_data, idxmonth, avgShift, swapYears = np.arange( futperiod[0], futperiod[1] + 1, 1 ) )
            # tmy3_fut  = fixRH( tmy3_fut )

        # New attributes for future file
        new_filetype      = f"f{self.filetype}"
        new_filename      = f"{self.location}_{new_filetype}_{futyear}_{futexp}_{model}.epw"
        new_years_in_file = new_data['Year'].unique().tolist()
        new_years_range   = ( futperiod[0], futperiod[ -1 ] )
        new_avgYear       = round( ( futperiod[ 0 ] + futperiod[ -1 ] ) / 2 )
        # Save the new file if requested
        if savedir is not None:
            new_filepath = os.path.join( savedir, new_filename )
        else:
            new_filepath = os.path.join( os.path.dirname( self.file_path ), new_filename )
        # Return a new EPWFile instance with the modified data
        return replace( self, data = new_data, file_path= new_filepath, filetype = new_filetype, filename = new_filename, years_in_file = new_years_in_file,
                        year_range = new_years_range, avgYear = new_avgYear )
        
        # self.filepath = f"{self.data_directory}/{self.location}/{self.filetype}/{self.filename}"
                    # Potentially IO
            # if filetype == 'TMY' or filetype == 'RMY':
            #     label        = 'SimpleShift_' + str( futureYear )
            # elif filetype == 'AMY':
            #     label       = 'SimpleShift_' + str( year[0] +  (futureYear - round( histperiod.mean() ) ) )
            # writeTmyFile( epwfile, indir, indir, locname, tmy3_fut,  label, filetype, model, futexp, futperiod )

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
        self.Nfiles           = len( self.files )
        if self.files == []:
            raise ValueError(f"No {self.obj_type} files found in the specified directory.")
        else:
            self.files = self.read_all_files()

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
        os.makedirs( f"{self.data_directory}/cmip6/{model}", exist_ok = True )
        # Download the files
        print( f"Downloading CMIP6 files for {model}... It might take a few minutes." )
        cmip6_files = list_svante_files( f"{self.online_directory}/cmip6/{model}", extension = ".nc" )
        for file in cmip6_files:
            file_url   = f"{self.online_directory}/cmip6/{model}/{file}"
            local_path = f"{self.data_directory}/cmip6/{model}/{file}"
            if not os.path.exists( local_path ):
                resp = requests.get( file_url )
                resp.raise_for_status()
                with open( local_path, "wb" ) as f:
                    f.write( resp.content )

    # TO BE FIXED/COMPLETED.         
    def futureShifts( self, params: dict, saveFlag: bool = False ):
        """
        Method to generate future shifted files for all EPWFile instances in the collection.
        Parameters
        ----------
        params : dict
            Dictionary containing parameters for future shift (model, member, futyear, futexp, pattern_exp, grid)
        saveFlag : bool, optional
            Whether to save the modified files (default: False)
        """
        # Download CMIP6 files if not already present
        model_dir = f"{self.data_directory}/cmip6/{ params['model'] }"
        if os.path.exists( model_dir ) is False:
            self.downloadCmip( params['model'] )
        else:
            print( f"CMIP6 files for {params['model']} already exist locally. Proceeding with future shift..." )
        future_files = []
        for epwfile in self.files:
            future_files.append( epwfile.with_futureShift( f"{self.data_directory}/cmip6", params, saveFlag ) )