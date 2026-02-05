# kranke - January 2026
# Script to define the main epwdata class for handling EPW files

import re
import os
import requests
import pandas       as     pd
import numpy        as     np
import copy
from   ioutils      import list_svante_files, read_nth_line
from   miscutils    import shift_tuple, swapMonthTmy
from   constants    import epw_colnames, months_labels
from   dataclasses  import dataclass, replace
from   cmip6utils   import CalcGlobalDT, getPatternCoefficients, calculateShift
from   pathlib      import Path

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
        # SSet the new filepath
        if savedir is not None:
            new_filepath = os.path.join( savedir, new_filename )
        else:
            new_filepath = os.path.join( os.path.dirname( self.file_path ), new_filename )
        # Assign new headers for future file
        month_year_pairs = new_data[['Month', 'Year']].drop_duplicates().sort_values(['Month','Year'])
        num_years        = futperiod[-1] - futperiod[0] + 1
        month_strs       = [f"{months_labels[row.Month-1]}={row.Year}" for row in month_year_pairs.itertuples(index=False)]
        new_comment1     = f'COMMENTS 1,"BC3 emulator - #years=[{num_years}] Period of Record={futperiod[0]}-{futperiod[-1]}; ' + "; ".join(month_strs) + '"'
        new_comment2     = f'COMMENTS 2,"{new_filetype.upper()} processed with BC3 Emulator -- pgiani@mit.edu for more info"'
        new_source       = f'BC3Emulator_{model}_{member}_{futexp}_{futyear}'

        # Return a new EPWFile instance with the modified data (Replace skips the post_init method)
        new_instance     = replace( self, data = new_data, file_path = new_filepath, filetype = new_filetype, filename = new_filename, years_in_file = new_years_in_file,
                        year_range = new_years_range, avgYear = new_avgYear, comment1 = new_comment1, comment2 = new_comment2, source = new_source ) 
        # Write out the file if savedir is specified
        if savedir is not None:
            new_instance.writeToFile( new_filepath )
        # Return the replaced instance
        return new_instance
            
    def writeToFile( self, output_path: str ):
        """
        Method to write the EPWFile data to a specified output path.
        Parameters
        ----------
        output_path : str
            The file path where the EPW data should be written.
        """
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
            self.data.drop( columns=['date', 'datetime'] ).to_csv( f, index = False, header = False )

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
        # Set amy_years if filetype is 'amy'
        if self.obj_type == 'amy':
            self.amy_years = [ file.avgYear for file in self.files ]
        
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
            params['model'] = 'CanESM5'
        # Download CMIP6 files if not already present
        model_dir = f"{self.data_directory}/cmip6/{ params['model'] }"
        if os.path.exists( model_dir ) is False:
            self.downloadCmip( params['model'] )
        else:
            print( f"CMIP6 files for {params['model']} already exist locally. Proceeding with future shift..." )
        future_files  = []
        yearsShift    = params.get( 'futyear', 2050 ) - round( sum( self.amy_years ) / len( self.amy_years ) )
        print( f"Average shift for each AMY file: {yearsShift} years" )
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