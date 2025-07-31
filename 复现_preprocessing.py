import time
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')



def read_data(data):
    print('The data is the 2015 borrower data of LendingClub opened on official website')
    df = pd.read_csv(data)
    df = df[df['issue_d'].str[-2:] == '15']
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
    trainData = data[(data['issue_d'] != 'Nov-2015') & (data['issue_d'] != 'Dec-2015')]
    testData = data[(data['issue_d'] == 'Nov-2015') | (data['issue_d'] == 'Dec-2015')]
    print('the train data shape is: ', trainData.shape)
    print('the test data shape is: ', testData.shape)
    return trainData, testData

def drop_afterloan_columns(data):
    after_col = ['pymnt_plan', 'collection_recovery_fee', 'recoveries', 'title',
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
    data = data.dropna(how='all', axis=1)
    df = data.dropna(how='all', axis=0)
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
    def int_rate(val):
        return round(float(str(val).replace('%','')) / 100, 4) if pd.notna(val) else -1

    def emp_length(val):
        if val == '10+ years':
            return 10
        elif val == '< 1 year':
            return 0
        elif val == 'n/a':
            return -1
        elif val == '1 year':
            return 1
        else:
            return float(str(val).replace('years', ''))

    def grade_value(val):
        mapping = {c: i + 1 for i, c in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}
        return val.map(mapping).fillna(0)

    def subgrade_value(val):
        mapping = {c: i + 1 for i, c in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G'])}
        return val.str[0].map(mapping).fillna(0) * 10 + val.str[1].astype(int)

    def ConvertDateStr(x):
        mth_dict = {
            c: i + 1
            for i, c in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        }
        if pd.isna(str(x)):
            return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-01', '%Y-%m')))
        else:
            yr = int(x[4:6])
            yr  = 2000 + yr if yr <= 17 else 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr, mth, 1)

    def days_long(val):
        now = datetime.datetime.now()
        delta = now - val
        return delta.days

    data['int_rate'] = data.loc[:, 'int_rate'].apply(int_rate)
    data['emp_length'] = data.loc[:, 'emp_length'].apply(emp_length)
    data['revol_util'] = data.loc[:, 'revol_util'].astype(str).apply(int_rate)
    data['grade'] = grade_value(data.loc[:, 'grade'])
    data['sub_grade'] = subgrade_value(data.loc[:, 'sub_grade'])
    data['earliest_cr_line'] = data.loc[:, 'earliest_cr_line'].apply(ConvertDateStr).apply(days_long)
    return data


def get_word_cat_ordered_continue_col(data):
    def class_feature(data):
        cat_col = data.select_dtypes(include=['object']).columns
        continue_col = [col for col in data.columns if col not in cat_col]
        return cat_col, continue_col

    word_col = ['zip_col', 'addr_state', 'emp_title']
    temp_col = ['mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mths_since_recent_bc',
                'mtgs_since_recent_inq', 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'bc_util', 'revol_util', 'dti']
    cat_col, value_col = class_feature(data)
    cat_col = [w for w in cat_col if w not in word_col]
    continue_col = [col for col in value_col if data[col].nunique() > 500]
    continue_col = [key for key in continue_col if key not in word_col + temp_col + ['emp_length']]
    ordered_col = [key for key in value_col if key not in continue_col]
    ordered_col = [key for key in ordered_col if key not in ['y', 'term']]
    ordered_col = ordered_col + ['emp_length']
    return word_col, cat_col, ordered_col, continue_col

def del_outlier_index(data, key):
    temp_index = data[data[key] > data[key].sort_values(ascending=False)[:10].min()].index
    return list(temp_index)