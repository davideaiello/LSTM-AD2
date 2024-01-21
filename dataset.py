import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import arguments
import pandas as pd
import torch
from torch.utils.data import DataLoader

args = arguments.parse_arguments()   

def read_folder_normal(dataset_folder, frequency):
    ROOTDIR_DATASET = dataset_folder
    filepaths_csv = [os.path.join(ROOTDIR_DATASET, f"rec{r}_20220811_rbtc_{1/frequency}s.csv") for r in [0, 2, 3, 4]]

    dfs = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepaths_csv]
    df = pd.concat(dfs)                           
    df = df.sort_index(axis=1)
    df.index = pd.to_datetime(df.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")

    columns_to_drop = [column for column in df.columns if "Abb" in column or "Temperature" in column]
    df.drop(["machine_nameKuka Robot_export_active_energy",                                             
            "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)

    df.drop(['time'], axis=1, inplace=True)
    X_train = df
    return X_train


def read_folder_collisions(dataset_folder, frequency):
    collisions = pd.read_excel(os.path.join(dataset_folder, "20220811_collisions_timestamp.xlsx"))
    collisions['Timestamp'] = collisions['Timestamp'] - pd.to_timedelta(2, 'h')

    start_col = collisions[collisions['Inizio/fine'] == "i"][['Timestamp']].rename(columns={'Timestamp': 'start'})
    end_col = collisions[collisions['Inizio/fine'] == "f"][['Timestamp']].rename(columns={'Timestamp': 'end'})

    start_col.reset_index(drop=True, inplace=True)
    end_col.reset_index(drop=True, inplace=True)

    df_collision = pd.concat([start_col, end_col], axis=1)
    
    filepath_csv_test = [os.path.join(dataset_folder, f"rec{r}_collision_20220811_rbtc_{1/frequency}s.csv") for r in [1, 5]]
    dfs_test = [pd.read_csv(filepath_csv, sep=";") for filepath_csv in filepath_csv_test]
    df_test = pd.concat(dfs_test)

    df_test = df_test.sort_index(axis=1)
    df_test.index = pd.to_datetime(df_test.time.astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")

    columns_to_drop = [column for column in df_test.columns if "Abb" in column or "Temperature" in column]     # e quindi non ritorna un
    df_test.drop(["machine_nameKuka Robot_export_active_energy",                                               # nuovo dataset
            "machine_nameKuka Robot_import_reactive_energy"] + columns_to_drop, axis=1, inplace=True)
    df_test['time'] = pd.to_datetime(df_test['time'].astype('datetime64[ms]'), format="%Y-%m-%dT%H:%M:%S.%f")
    X_collisions = df_test.drop(['time'], axis=1, inplace=False)
    return df_collision, X_collisions, df_test

def preprocess_data(data, pipeline=None, train=True):
    if train:
        X_train = data
        pipeline = Pipeline([
            ('scaler', preprocessing.MinMaxScaler()),
            ('selector', VarianceThreshold(threshold=0.001))
            ])
        X_train_transformed = pipeline.fit_transform(X_train)
        return X_train_transformed, pipeline
    else:
        X_collisions = data
        X_collisions_transformed = pipeline.transform(X_collisions)
        return X_collisions_transformed

def split_data(ts, split=0.9, df_test=None):
    split_at = int(len(ts) *split)
    train_ts = ts[:split_at]
    valid_ts = ts[split_at:]
    train_ds = TimeSeries_train_val(train_ts, window_length=args.window_size, prediction_length=args.prediction_length)
    valid_ds = TimeSeries_estimate_eval(valid_ts, window_length=args.window_size, prediction_length=args.prediction_length)
    if df_test is not None:
        df_col = df_test.iloc[:split_at]
        df_val = df_test.iloc[split_at:]
        return DataLoader(train_ds, batch_size=args.infer_batch_size), DataLoader(valid_ds, batch_size=args.infer_batch_size), df_col, df_val
    else:
        return DataLoader(train_ds, batch_size=args.batch_size), DataLoader(valid_ds, batch_size=args.infer_batch_size)

def return_dataloader(ts):
    ds = TimeSeries_estimate_eval(ts, window_length=args.window_size, prediction_length=args.prediction_length)
    return DataLoader(ds, batch_size=args.infer_batch_size)

class TimeSeries_train_val(Dataset):
    def __init__(self, X, window_length, prediction_length):
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length
        self.l = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.l

    def __getitem__(self, index):
        end_idx = index + self.window_length
        x = self.X[index:end_idx]
        y = self.X[end_idx:end_idx+self.l, :]
        return x, y
    
    
class TimeSeries_estimate_eval(Dataset):
    def __init__(self, X, window_length, prediction_length):
        self.X = torch.from_numpy(X).float()
        self.window_length = window_length
        self.l = prediction_length

    def __len__(self):
        return self.X.shape[0] - (self.window_length - 1) - self.l-1

    def __getitem__(self, index):
        end_idx = index + self.window_length
        x = []
        for p in range(self.l):
            x.append(self.X[index + p : p + end_idx])                   # qui prende tutte le finestre w
        x = torch.cat([torch.unsqueeze(item, 0) for item in x], dim=0) # concatena mettendo la nuova dimensione, self.prediction_length, davanti
        y = self.X[end_idx + self.l]
        y = y.repeat(self.l)
        return x, y
    

    




