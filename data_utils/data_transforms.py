import torch


class DataTransform:
    DERIVED_FEATURES = {
        "Return",
        "LogReturn",
        "HLRange",
        "OCChange",
        "VolumeChange",
        "RollingVolatility_7",
    }

    def __init__(self, is_train, use_volume=False, additional_features=None):
        self.is_train = is_train
        self.additional_features = list(additional_features or [])
        self.base_keys = ["Timestamp", "Open", "High", "Low", "Close"]
        if use_volume:
            self.base_keys.append("Volume")
        self.base_keys.extend(self.additional_features)
        print(self.base_keys)

    def _get_keys(self, window):
        keys = list(self.base_keys)
        if "Timestamp_orig" in window.keys():
            keys.append("Timestamp_orig")
        return keys

    def _get_values(self, window, key):
        if key in window.keys():
            return torch.tensor(window.get(key).tolist(), dtype=torch.float32)
        if key not in self.DERIVED_FEATURES:
            raise KeyError(f"Feature '{key}' is neither a data column nor a supported derived feature.")

        close = torch.tensor(window.get("Close").tolist(), dtype=torch.float32)
        open_price = torch.tensor(window.get("Open").tolist(), dtype=torch.float32)
        high = torch.tensor(window.get("High").tolist(), dtype=torch.float32)
        low = torch.tensor(window.get("Low").tolist(), dtype=torch.float32)
        previous_close = torch.roll(close, shifts=1)
        previous_close[0] = close[0]
        close_return = (close - previous_close) / previous_close.clamp_min(1e-6)

        if key == "Return":
            return close_return
        if key == "LogReturn":
            return torch.log(close.clamp_min(1e-6) / previous_close.clamp_min(1e-6))
        if key == "HLRange":
            return (high - low) / close.clamp_min(1e-6)
        if key == "OCChange":
            return (close - open_price) / open_price.clamp_min(1e-6)
        if key == "RollingVolatility_7":
            values = []
            for idx in range(close_return.numel()):
                start = max(0, idx - 6)
                values.append(close_return[start : idx + 1].std(unbiased=False))
            return torch.stack(values)

        volume = torch.tensor(window.get("Volume").tolist(), dtype=torch.float32)
        previous_volume = torch.roll(volume, shifts=1)
        previous_volume[0] = volume[0]
        return (volume - previous_volume) / previous_volume.clamp_min(1.0)

    def __call__(self, window):
        data_list = []
        output = {}
        for key in self._get_keys(window):
            if key in {"Timestamp", "Timestamp_orig"}:
                data = torch.tensor(window.get(key).tolist(), dtype=torch.float64)
                feature_data = data.to(torch.float32)
            else:
                data = self._get_values(window, key)
                feature_data = data
            if key == "Volume":
                data /= 1e9
                feature_data /= 1e9
            output[key] = data[-1]
            output[f"{key}_old"] = data[-2]
            if key == "Timestamp_orig":
                continue
            data_list.append(feature_data[:-1].reshape(1, -1))
        output["features"] = torch.cat(data_list, 0)
        return output
