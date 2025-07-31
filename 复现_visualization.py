from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
plt.style.use('bmh')


def plot_label(data, col):
    temp = data.groupby('y').count().iloc[:, 0]
    bar_data = {'normal': temp[0], 'overdue': temp[1]}
    values = list(bar_data.values())
    plt.bar(range(2), values)
    plt.xticks((0,1), ('normal', 'overdue'))
    plt.title('borrower loan status distribution')
    plt.text(0.45, 500000, r'normal: overdue mostly equal to 20', color='black')
    plt.text(0.45, 400000, r'unbalanced dataset', color='black')
    plt.show()


def plot_cat(data, key):
    plot_data = data[[key, 'y']]
    plt.figure(figsize=(8, 6))
    params = {
        'home_ownership': ['ANY', 'RENT', 'MORTGAGE', 'OWN'],
        'verification': ['Source Verified', 'Not Verified', 'Verified'],
        'initial_list_status': ['w', 'f'],
        'grade': [1, 2, 3, 4, 5, 6, 7],
        'emp_length': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.027167, 8.0, 9.0, 10.0, 7.0],
        'purpose': ['home_improvement', 'medical', 'educational', 'other', 'debt_consolidation', 'vacation', 'house',
                    'wedding', 'major_purpose', 'moving', 'car', 'small_business', 'renewable_energy', 'credit_card'],
        'issue_d': ['Oct-20']
    }