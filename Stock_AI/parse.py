import pandas as pd
import os

data_folder = "./Data/"
output_folder = "./Parsed_Data/"

atr_len = 14
atr_short = 1.
atr_long = 2.

def parse(cur_file_name):
    df = pd.read_csv(f"{data_folder}{cur_file_name}", sep="\t")

    df['true-range'] = df['High'] - df['Low']
    df['ATR'] = df.loc[:,'true-range'].rolling(window=atr_len).mean()

    df['high_bar'] = df['Close'] + atr_long * df['ATR']
    df['low_bar'] = df['Close'] - atr_long * df['ATR']
    df['high_close'] = df['Close'] + atr_short * df['ATR']
    df['low_close'] = df['Close'] - atr_short * df['ATR']

    df = df[atr_len:]

    df.to_csv(f"{output_folder}{cur_file_name}")
    print(cur_file_name)



files = os.listdir(data_folder)
for file in files:
    parse(file)