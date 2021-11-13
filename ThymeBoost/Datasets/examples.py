# -*- coding: utf-8 -*-
import pandas as pd


def get_data(dataset_name):
    df = pd.read_csv(fr"..\{dataset_name}.csv")
    return df
