# -*- coding: utf-8 -*-
import pandas as pd
from pathlib import Path




def get_data(dataset_name):
    file_path = str(Path.cwd().resolve())
    df = pd.read_csv(fr"{file_path}\ThymeBoost\Datasets\{dataset_name}.csv")
    return df
