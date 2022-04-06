This module contains miscellaneous utilty scripts and resources that are used across the project.
- The pems module contains all functions for downloading files from the PEMS site. Make a python script in this directory called ```creds.py``` with variables "username" and "password" to store your login credentials.
- The script ```data_loader.py``` scrapes the PEMS website and loads the data into ```pems/```.
- The notebooks ```station_data_extractor.ipynb``` and ```station_meta_extractor.ipynb``` lightly processes and load the PEMS data into pickle files for further processing and EDA.
- The notebook ```data_pkl_to_csv.ipynb``` converts all processed data as pickle files to csvs.
- The notebook ```set_env_vars.ipynb``` loads environment variables (mostly training parameters) into ```../model/env.yaml```.
- The script ```utils.py``` contains utility functions like data loading, data processing, and cross validation which are used in other notebooks.
- The directory ```STGCN/``` contains the implementation code developed by researchers at Peking University who developed the architecture. We leverage this to train our own STGCN and evaluate its performance.






