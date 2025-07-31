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
import 复现_visualization as vs
# import scorecard_functions_V3 as sf
# import feature_bin as fb
# import helper as lp


if __name__ == '__main__':

    data = pc.read_data('loan.csv')
    print(data.groupby('loan_status').size())
    # trainData, testData = pc.split_train_test(data)
    #
    # #choose features and samples
    # df = pc.drop_afterloan_columns(trainData)
    # df = pc.drop_unique1_col(df)
    # df = pc.drop_missingmore60_col(df)
    # df = pc.drop_row_col_miss(df)
    # df = pc.drop_90samevalue_col(df)
    #
    # print('=====')
    # df = pc.get_label(df, 'loan_status')
    # df = pc.string_to_value(df)
    # print('now, print top 5 row data: ')
    # print(df.head())
    #
    # #outlier preprocessing
    # word_col, cat_col, ordered_col, continue_col = pc.get_word_cat_ordered_continue_col(df)
    # ordered_col_outlier = [
    #     w for w in ordered_col
    #     if w not in ['grade', 'sub_grade', 'earliest_cr_line', 'int_rate','inq_last_6mths', 'emp_length', 'dti',
    #                  'percent_bc_gt_75','pub_rec_bankruptcies']
    # ]
    # all_outlier = ordered_col_outlier + continue_col
    # all_outlier_index = [pc.del_outlier_index(df, key) for key in all_outlier]
    # outlier_index = list(set(list(itertools.chain(*all_outlier_index))))
    # df = df.drop(outlier_index, axis=0)
    # print('after delete outliers, the data shape now is: ', df.shape)
    #
    # f = open('df.pkl', 'wb')
    # pickle.dump(df, f)
    # with open('df.pkl', 'rb') as f:
    #     df = pickle.load(f)
    #
    # #feature visualization
    # vs.plot_label(df, 'y')
    #
    # word_col, cat_col, ordered_col, continue_col = pc.get_word_cat_ordered_continue_col(df)
    # for key in [w for w in cat_col]:
    #     vs.plot_cat(df, key)