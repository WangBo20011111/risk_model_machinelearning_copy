import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')



def read_data(data):
    print('The data is the third quarter of 2017 borrower data of LendingClub opened on official website')
    df = pd.read_csv(data)
    print(sorted(df.columns))
    df = df[df['term'] == ' 36 months']
    print('\n')
    print('top 5 line of data is :\n', df.head(5))
    print('\n')
    print('data statistic information is', df.describe())
    print('\n')
    print('all data shape is ', df.shape)
    return df

def split_train_test(data):
    trainData = data[(data['issue_d'] != 'Nov-2015') & (data['issue_id'] != 'Dec-2015')]
    testData = data[(data['issue_d'] == 'Nov-2015') | (data['issue_d'] == 'Dec-2015')]
    print('the train data shape is: ', trainData.shape)
    print('the test data shape is: ', testData.shape)
    return trainData, testData

def drop_afterloan_columns(data):
    after_col = ['pymnt_plan', 'collection_recovery_fee', 'recoveries', 'hardship_flag', 'title',
                 'out_prncp_inv', 'out_prncp', 'total_rec_prncp', 'last_pymnt_amnt', 'last_pymnt_d',
                 'last_credit_pull_d', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'term']
    df = data.drop(after_col, axis=1)
    print('after drop after loan data, the data shape is ', df.shape)
    return df

def drop_unique1_col(data):
    cols = data.nunique()[data.nunique() > 1].index.tolist()
    df = data.loc[:, cols]
    print('after drop only one value columns, the data shape is ', df.shape)
    return df

def drop_missingmore60_col(data):
    miss_60_col = data.isnull().sum()[data.isnull().sum() >= 0.40 * data.shape[0]].index
    df = data.drop(miss_60_col, axis=1)
    print('after drop missing greater than 60% columns, the data shape is ', df.shape)
    return df

def drop_row_col_miss(data):
    data = data.drop(how='all', axis=1)
    df = data.drop(how='all', axis=0)
    return df

def drop_90samevalue_col(data):
    colum = data.columns
    per = pd.DataFrame(colum, index=colum)
    max_valuecounts = []
    for col in colum:
        max_valuecounts.append(data[col].value_counts().max())
    per['mode'] = max_valuecounts
    per['percentile'] = per['mode'] / data.shape[0]
    same_value_col = per[per.sort_values(by='percentile', ascending=False)['percentile'] > 0.9].index
    df = data.drop(same_value_col, axis=1)
    print('after delete 90% values same in one column, the data shape is ', df.shape)
    return df

def label_transe(val):
    if val == 'Charged Off':
        return 1
    elif val == 'Fully Paid' or val == 'Current':
        return 0
    else:
        return -1

def get_label(data, label):
    data['y'] = data[label].apply(label_transe)
    data = data[((data['y'] != 2) & (data['y'] != -1))].drop([label], axis=1)
    return data

def string_to_value(data):
    data['int_rate'] = data.loc[:, 'int_rate'].apply(int_rate)
    data['emp_length'] = data.loc[:, 'emp_length'].apply(emp_length)
    data['revol_util'] = data.loc[:, 'revol_util'].astype(str).apply(int_rate)
    data['grade'] = grade_value(data.loc[:, 'grade'])
    data['sub_grade'] = subgrade_value(data.loc[:, 'sub_grade'])
    data['earliest_cr_line'] = data.loc[:, 'earliest_cr_line'].apply(ConvertDateStr).apply(days_long)
    return data