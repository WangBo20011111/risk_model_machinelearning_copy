import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import missingno as msno
import datetime
import warnings
warnings.filterwarnings('ignore')

import 复现_preprocessing as pc
# import scorecard_functions_V3 as sf
# import feature_bin as fb
# import visualization as vs
# import helper as lp


if __name__ == '__main__':

    data = pc.read_data('loan.csv')
    trainData, testData = pc.split_train_test(data)

    df = pc.drop_afterloan_columns(trainData)
    df = pc.drop_unique1_col(df)
    df = pc.drop_missingmore60_col(df)
    df = pc.drop_row_col_miss(df)
    df = pc.drop_90samevalue_col(df)

    print('=====')
    df = pc.get_label(df, 'loan_status')
    df = pc.string_to_value(df)
    print('now, print top 5 row data: ')
    df.head()
