import pickle
import pandas as pd

def load_processed_data(path_to_data, method='pickle'):
    if method == 'pickle':
        with open(f'{path_to_data}/adj_mat.dat', 'rb')  as f:
            adj_mat = pickle.load(f)

        with open(f'{path_to_data}/adj_mat_ind_station_mapper.dat', 'rb') as f:
            ind_station_mapper = pickle.load(f)

        with open(f'{path_to_data}/speeds.dat', 'rb') as f:
            speed_df = pickle.load(f)
    elif method == 'csv':
        adj_mat = pd.read_csv(f'{path_to_data}/csvs/adj_mat.csv', index_col=0).values
        ind_station_mapper = pd.read_csv(f'{path_to_data}/csvs/adj_mat_ind_station_mapper.csv', index_col=0).set_index('adj_mat_ind').to_dict()['station_ind']
        speed_df = pd.read_csv(f'{path_to_data}/csvs/speeds.csv', index_col=0)
        speed_df.index = pd.to_datetime(speed_df.index)
    else:
        raise Exception('Method must be pickle or csv.')
    return adj_mat, ind_station_mapper, speed_df