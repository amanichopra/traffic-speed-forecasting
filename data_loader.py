import sys
sys.path.append('./utils')
import pandas as pd
from pems.handler import PeMSHandler
import creds

START_YEAR = 2021
END_YEAR = 2021
DISTRICTS = [7]
FILE_TYPES = ['station_5min', 'meta']
MONTHS = 'all'
SAVE_PATH = './data/pems'

DISTRICTS = [str(d) for d in DISTRICTS]
if MONTHS == 'all': MONTHS = None
    
# Connect to PeMS
pems = PeMSHandler(username=creds.username, password=creds.password)
pems.download_files(start_year=START_YEAR, end_year=END_YEAR, districts=DISTRICTS, 
                    file_types=FILE_TYPES, months=MONTHS, save_path=SAVE_PATH)