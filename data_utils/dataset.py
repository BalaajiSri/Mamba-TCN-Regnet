import os
import torch
import time
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import io_tools
from datetime import datetime
from utils.io_tools import load_config_from_yaml

    
class CMambaDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        split,
        window_size,
        transform,
    ):

        self.data = data
        self.transform = transform
        self.window_size = window_size
            
        print('{} data points loaded as {} split.'.format(len(self), split))

    def __len__(self):
        return max(0, len(self.data) - self.window_size - 1)

    def __getitem__(self, i: int):
        sample = self.data.iloc[i: i + self.window_size + 1]
        sample = self.transform(sample)
        return sample
    
class DataConverter:
    DERIVED_FEATURES = {
        "Return",
        "LogReturn",
        "HLRange",
        "OCChange",
        "VolumeChange",
        "RollingVolatility_7",
    }

    def __init__(self, config) -> None:
        self.config = config
        self.root = config.get('root')
        self.jumps = config.get('jumps')
        self.date_format = config.get('date_format', "%Y-%m-%d")
        self.additional_features = config.get('additional_features', [])
        self.end_date = config.get('end_date')
        self.data_path = config.get('data_path')
        self.start_date = config.get('start_date')
        split_tag = config.get('split_name', 'default')
        self.folder_name = f'{self.start_date}_{self.end_date}_{self.jumps}_{split_tag}'
        self.file_path = f'{self.root}/{self.folder_name}'

    
    def process_data(self):
        data, start, stop = self.load_data()
        new_df = {}
        for key in ['Timestamp', 'High', 'Low', 'Open', 'Close', 'Volume']:
            new_df[key] = []
        for key in self.additional_features:
            if key in self.DERIVED_FEATURES:
                continue
            new_df[key] = []
        for i in tqdm(range(start, stop - self.jumps // 60 + 1, self.jumps)):
            high, low, open, close, vol = self.merge_data(data, i, self.jumps)
            additional_features = self.merge_additional(data, i, self.jumps)
            if high is None:
                continue
            new_df.get('Timestamp').append(i)
            new_df.get('High').append(high)
            new_df.get('Low').append(low)
            new_df.get('Open').append(open)
            new_df.get('Close').append(close)
            new_df.get('Volume').append(vol)
            for key in additional_features.keys():
                new_df.get(key).append(additional_features.get(key))

        df = pd.DataFrame(new_df)

        return df

    def get_data(self):
        tmp = '----'
        data_path = f'{self.file_path}/{tmp}.csv'
        yaml_path = f'{self.file_path}/config.yaml'
        if os.path.isfile(data_path.replace(tmp, 'train')) and os.path.isfile(yaml_path):
            cached_config = io_tools.load_config_from_yaml(yaml_path)
            if cached_config == self.config:
                train = pd.read_csv(data_path.replace(tmp, 'train'), index_col=0)
                val = pd.read_csv(data_path.replace(tmp, 'val'), index_col=0)
                test = pd.read_csv(data_path.replace(tmp, 'test'), index_col=0)
                return train, val, test

        if os.path.isfile(data_path.replace(tmp, 'train')) and not os.path.isfile(yaml_path):
            train = pd.read_csv(data_path.replace(tmp, 'train'), index_col=0)
            val = pd.read_csv(data_path.replace(tmp, 'val'), index_col=0)
            test = pd.read_csv(data_path.replace(tmp, 'test'), index_col=0)
            return train, val, test
        
        df = self.process_data()

        train, val, test = self.split(df)

        if not os.path.isdir(self.file_path):
            os.makedirs(self.file_path, exist_ok=True)

        train.to_csv(data_path.replace('----', 'train'))
        val.to_csv(data_path.replace('----', 'val'))
        test.to_csv(data_path.replace('----', 'test'))
        io_tools.save_yaml(self.config, yaml_path)
        return train, val, test
    

    def split(self, data):
        if self.config.get('train_ratio') is not None:
            total = len(data)
            n_train = int(self.config.get('train_ratio') * total)
            n_test = int(self.config.get('test_ratio') * total)
            total = list(range(total))
            random.shuffle(total)
        
            train = sorted(total[: n_train])
            val = sorted(total[n_train: -n_test])
            test = sorted(total[-n_test: ])
            train = data.iloc[train]
            val = data.iloc[val]
            test = data.iloc[test]
        else:
            tmp_dict = {}
            for key in ['train', 'val', 'test']:
                start, stop = self.config.get(f'{key}_interval')
                start = self.generate_timestamp(start)
                stop = self.generate_timestamp(stop)
                tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
                tmp = tmp[tmp['Timestamp'] < stop].reset_index(drop=True)
                tmp_dict[key] = tmp
            train = tmp_dict.get('train')
            val = tmp_dict.get('val')
            test = tmp_dict.get('test')
        return train, val, test


    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'Timestamp' not in df.keys():
            dates = df.get('Date').to_list()
            df['Timestamp'] = [self.generate_timestamp(x) for x in dates]
        if self.start_date is None:
            self.start_date = self.convert_timestamp(min(list(df.get('Timestamp')))).strftime(self.date_format)
            # raise ValueError(self.start_date)
        if self.end_date is None:
            self.end_date = self.convert_timestamp(max(list(df.get('Timestamp'))) + self.jumps).strftime(self.date_format)
        start = self.generate_timestamp(self.start_date)
        stop = self.generate_timestamp(self.end_date)
        df = df[df['Timestamp'] >= start].reset_index(drop=True)
        df = df[df['Timestamp'] < stop].reset_index(drop=True)
        final_day = self.generate_timestamp(self.end_date)
        return df, start, final_day
    
    def merge_additional(self, data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return {key: None for key in self.additional_features if key not in self.DERIVED_FEATURES}
        row = tmp.iloc[-1]
        results = {}
        for key in self.additional_features:
            if key in self.DERIVED_FEATURES:
                continue
            results[key] = float(row.get(key))
        return results
    
    def generate_timestamp(self, date):
        return int(time.mktime(datetime.strptime(date, self.date_format).timetuple()))
    
    @staticmethod
    def merge_data(data, start, jump):
        tmp = data[data['Timestamp'] >= start].reset_index(drop=True)
        tmp = tmp[tmp['Timestamp'] < start + jump].reset_index(drop=True)
        if len(tmp) == 0:
            return None, None, None, None, None
        _, high, low, open, close, _ = DataConverter.get_row_values(tmp.iloc[0])
        vol = 0
        for row in tmp.iterrows():
            _, h, l, _, close, v = DataConverter.get_row_values(row[1])
            high = max(high, h)
            low = min(low, l)
            vol += v
        return high, low, open, close, vol
    
    @staticmethod
    def get_row_values(row):
        ts = int(row.get('Timestamp'))
        high = float(row.get('High'))
        low = float(row.get('Low'))
        open = float(row.get('Open'))
        close = float(row.get('Close'))
        vol = float(row.get('Volume'))
        return ts, high, low, open, close, vol

    @staticmethod
    def convert_timestamp(timestamp):
        return datetime.fromtimestamp(timestamp)


class NegControlDataset(torch.utils.data.Dataset):
    """Wraps CMambaDataset to corrupt targets for negative-control experiments.

    shuffled_labels  – features from sample i paired with targets from perm[i].
                       Expected result: direction accuracy ≈ 33 % (3-class chance).
    lag_mismatch     – features from sample i paired with targets from sample i+1.
                       Detects look-ahead leakage; well-built models should collapse.

    Only the training split is corrupted.  Val/test loaders use the original dataset.
    """

    TARGET_KEYS = ("Close", "Close_old")

    def __init__(
        self,
        base_dataset: CMambaDataset,
        mode: str,
        seed: int = 0,
    ) -> None:
        if mode not in ("shuffled_labels", "lag_mismatch"):
            raise ValueError(f"Unknown negcontrol mode '{mode}'. Choose 'shuffled_labels' or 'lag_mismatch'.")
        self.base = base_dataset
        self.mode = mode
        n = len(base_dataset)
        rng = torch.Generator().manual_seed(seed)
        self._perm = torch.randperm(n, generator=rng).tolist()

    def __len__(self) -> int:
        if self.mode == "lag_mismatch":
            return max(0, len(self.base) - 1)
        return len(self.base)

    def __getitem__(self, i: int):
        sample = dict(self.base[i])
        if self.mode == "shuffled_labels":
            target_sample = self.base[self._perm[i]]
        else:  # lag_mismatch: predict one step further ahead than intended
            target_sample = self.base[i + 1]
        for key in self.TARGET_KEYS:
            if key in target_sample:
                sample[key] = target_sample[key]
        return sample
