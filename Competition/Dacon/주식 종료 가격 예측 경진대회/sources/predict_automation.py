import os
import multiprocessing
import copy
import pickle
import warnings
from datetime import datetime, timedelta
from time import time, sleep, mktime
from matplotlib import font_manager as fm, rc, rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

import numpy as np
from numpy import array, nan, random as rnd, where
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from scipy.stats import f_oneway

from sklearn import preprocessing as prep
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split as tts, GridSearchCV as GridTuner, StratifiedKFold, KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.pipeline import make_pipeline

from sklearn import linear_model as lm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn import neighbors as knn
from sklearn import ensemble

# ===== tensorflow =====
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics as tf_metrics
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa
import keras_tuner as kt
from keras_tuner import HyperModel

# # ===== NLP =====
# from selenium import webdriver
# from konlpy.tag import Okt
# from KnuSentiLex.knusl import KnuSL

# ===== timeseries =====
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.preprocessing import timeseries_dataset_from_array as make_ts_tensor

warnings.filterwarnings(action='ignore')
rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# font setting
font_path = 'myfonts/NanumSquareB.ttf'
font_obj = fm.FontProperties(fname=font_path, size=12).get_name()
rc('font', family=font_obj)

# %reset -f

# ===== utility functions =====
# label encoding for categorical column with excepting na value
def which(bool_list):
    idx_array = where(bool_list)[0]
    return idx_array[0] if len(idx_array) == 1 else idx_array
def easyIO(x=None, path=None, op="r"):
    tmp = None
    if op == "r":
        with open(path, "rb") as f:
            tmp = pickle.load(f)
        return tmp
    elif op == "w":
        tmp = {}
        print(x)
        if type(x) is dict:
            for k in x.keys():
                if "MLP" in k:
                    tmp[k] = {}
                    for model_comps in x[k].keys():
                        if model_comps != "model":
                            tmp[k][model_comps] = copy.deepcopy(x[k][model_comps])
                    print(F"INFO : {k} model is removed (keras)")
                else:
                    tmp[k] = x[k]
        if input("Write [y / n]: ") == "y":
            with open(path, "wb") as f:
                pickle.dump(tmp, f)
            print("operation success")
        else:
            print("operation fail")
    else:
        print("Unknown operation type")
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]
def findIdx(data_x, col_names):
    return [int(i) for i, j in enumerate(data_x) if j in col_names]
def orderElems(for_order, using_ref):
    return [i for i in using_ref if i in for_order]
# concatenate by row
def ccb(df1, df2):
    if type(df1) == series:
        tmp_concat = series(pd.concat([dataframe(df1), dataframe(df2)], axis=0, ignore_index=True).iloc[:,0])
        tmp_concat.reset_index(drop=True, inplace=True)
    elif type(df1) == dataframe:
        tmp_concat = pd.concat([df1, df2], axis=0, ignore_index=True)
        tmp_concat.reset_index(drop=True, inplace=True)
    elif type(df1) == np.ndarray:
        tmp_concat = np.concatenate([df1, df2], axis=0)
    else:
        print("Unknown Type: return 1st argument")
        tmp_concat = df1
    return tmp_concat
def change_width(ax, new_value):
    for patch in ax.patches :
        current_width = patch.get_width()
        adj_value = current_width - new_value
        # we change the bar width
        patch.set_width(new_value)
        # we recenter the bar
        patch.set_x(patch.get_x() + adj_value * .5)
def week_of_month(date):
    month = date.month
    week = 0
    while date.month == month:
        week += 1
        date -= timedelta(days=7)
    return week
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def dispPerformance(result_dic, result_metrics):
    perf_table = dataframe(columns=result_metrics)
    for k, v in result_dic.items():
        perf_table = pd.concat([perf_table, v["performance"]], ignore_index=True, axis=0)
    print(perf_table)
    return perf_table

class MyLabelEncoder:
    def __init__(self, preset={}):
        # dic_cat format -> {"col_name": {"value": replace}}
        self.dic_cat = preset
    def fit_transform(self, data_x, col_names):
        tmp_x = copy.deepcopy(data_x)
        for i in col_names:
            # type check
            if not ((tmp_x[i].dtype.name == "object") or (tmp_x[i].dtype.name == "category")):
                print(F"WARNING : {i} is not object or category")
            # if key is not in dic, update dic
            if i not in self.dic_cat.keys():
                tmp_dic = dict.fromkeys(sorted(set(tmp_x[i]).difference([nan])))
                label_cnt = 0
                for j in tmp_dic.keys():
                    tmp_dic[j] = label_cnt
                    label_cnt += 1
                self.dic_cat[i] = tmp_dic
            # transform value which is not in dic to nan
            tmp_x[i] = tmp_x[i].astype("object")
            conv = tmp_x[i].replace(self.dic_cat[i])
            for conv_idx, j in enumerate(conv):
                if j not in self.dic_cat[i].values():
                    conv[conv_idx] = nan
            # final return
            tmp_x[i] = conv.astype("float")
        return tmp_x
    def transform(self, data_x):
        tmp_x = copy.deepcopy(data_x)
        for i in list(self.dic_cat.keys()):
            if not ((tmp_x[i].dtype.name == "object") or (tmp_x[i].dtype.name == "category")):
                print(F"WARNING : {i} is not object or category")
            # transform value which is not in dic to nan
            tmp_x[i] = tmp_x[i].astype("object")
            conv = tmp_x[i].replace(self.dic_cat[i])
            for conv_idx, j in enumerate(conv):
                if j not in self.dic_cat[i].values():
                    conv[conv_idx] = nan
            # final return
            tmp_x[i] = conv.astype("float")
        return tmp_x
    def clear(self):
        self.dic_cat = {}
class MyOneHotEncoder:
    def __init__(self, label_preset={}):
        self.dic_cat = {}
        self.label_preset = label_preset
    def fit_transform(self, data_x, col_names):
        tmp_x = dataframe()
        for i in data_x:
            if i not in col_names:
                tmp_x = pd.concat([tmp_x, dataframe(data_x[i])], axis=1)
            else:
                if not ((data_x[i].dtype.name == "object") or (data_x[i].dtype.name == "category")):
                    print(F"WARNING : {i} is not object or category")
                self.dic_cat[i] = OneHotEncoder(sparse=False, handle_unknown="ignore")
                conv = self.dic_cat[i].fit_transform(dataframe(data_x[i])).astype("int")
                col_list = []
                for j in self.dic_cat[i].categories_[0]:
                    if i in self.label_preset.keys():
                        for k, v in self.label_preset[i].items():
                            if v == j:
                                col_list.append(str(i) + "_" + str(k))
                    else:
                        col_list.append(str(i) + "_" + str(j))
                conv = dataframe(conv, columns=col_list)
                tmp_x = pd.concat([tmp_x, conv], axis=1)
        return tmp_x
    def transform(self, data_x):
        tmp_x = dataframe()
        for i in data_x:
            if not i in list(self.dic_cat.keys()):
                tmp_x = pd.concat([tmp_x, dataframe(data_x[i])], axis=1)
            else:
                if not ((data_x[i].dtype.name == "object") or (data_x[i].dtype.name == "category")):
                    print(F"WARNING : {i} is not object or category")
                conv = self.dic_cat[i].transform(dataframe(data_x[i])).astype("int")
                col_list = []
                for j in self.dic_cat[i].categories_[0]:
                    if i in self.label_preset.keys():
                        for k, v in self.label_preset[i].items():
                            if v == j: col_list.append(str(i) + "_" + str(k))
                    else:
                        col_list.append(str(i) + "_" + str(j))
                conv = dataframe(conv, columns=col_list)
                tmp_x = pd.concat([tmp_x, conv], axis=1)
        return tmp_x
    def clear(self):
        self.dic_cat = {}
        self.label_preset = {}
class MyKNNImputer:
    def __init__(self, k=5):
        self.imputer = KNNImputer(n_neighbors=k)
        self.cat_dic = {}
    def fit_transform(self, x, y, cat_vars=None):
        naIdx = dict.fromkeys(cat_vars)
        for i in cat_vars:
            self.cat_dic[i] = diff(list(sorted(set(x[i]))), [nan])
            naIdx[i] = list(which(array(x[i].isna()))[0])
        x_imp = dataframe(self.imputer.fit_transform(x, y), columns=x.columns)

        # if imputed categorical value are not in the range, adjust the value
        for i in cat_vars:
            x_imp[i] = x_imp[i].apply(lambda x: int(round(x, 0)))
            for j in naIdx[i]:
                if x_imp[i][j] not in self.cat_dic[i]:
                    if x_imp[i][j] < self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][0]
                    elif x_imp[i][j] > self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][len(self.cat_dic[i]) - 1]
        return x_imp
    def transform(self, x):
        naIdx = dict.fromkeys(self.cat_vars)
        for i in self.cat_dic.keys():
            naIdx[i] = list(which(array(x[i].isna())))
        x_imp = dataframe(self.imputer.transform(x), columns=x.columns)

        # if imputed categorical value are not in the range, adjust the value
        for i in self.cat_dic.keys():
            x_imp[i] = x_imp[i].apply(lambda x: int(round(x, 0)))
            for j in naIdx[i]:
                if x_imp[i][j] not in self.cat_dic[i]:
                    if x_imp[i][j] < self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][0]
                    elif x_imp[i][j] > self.cat_dic[i][0]:
                        x_imp[i][naIdx[i]] = self.cat_dic[i][len(self.cat_dic[i]) - 1]
        return x_imp
    def clear(self):
        self.imputer = None
        self.cat_dic = {}
folder_path = "projects/asiae_project3/dacon_stockprediction/"

# ===== task specific =====
from pykrx import stock
def getBreakthroughPoint(df, col1, col2, patient_days, fill_method="fb"):
    '''
    :param df: dataframe (including col1, col2)
    :param col1: obj
    :param col2: obj moving average
    :param patient_days: patient days detected as breakthrough point
    :return: signal series
    '''
    sigPrice = []
    flag = -1  # A flag for the trend upward/downward

    for i in range(0, len(df)):
        if df[col1][i] > df[col2][i] and flag != 1:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("buy")
                flag = 1
            else:
                if (tmp.iloc[1:] > tmp.iloc[0]).all():
                    sigPrice.append("buy")
                    flag = 1
                else:
                    sigPrice.append(nan)
        elif df[col1][i] < df[col2][i] and flag != 0:
            tmp = df['Close'][i:(i + patient_days + 1)]
            if len(tmp) == 1:
                sigPrice.append("sell")
                flag = 0
            else:
                if (tmp.iloc[1:] < tmp.iloc[0]).all():
                    sigPrice.append("sell")
                    flag = 0
                else:
                    sigPrice.append(nan)
        else:
            sigPrice.append(nan)

    sigPrice = series(sigPrice)
    for idx, value in enumerate(sigPrice):
        if not isna(value):
            if value == "buy":
                sigPrice.iloc[1:idx] = "sell"
            else:
                sigPrice.iloc[1:idx] = "buy"
            break
    # if fill_method == "bf":
    #
    # elif fill_method == ""
    sigPrice.ffill(inplace=True)
    return sigPrice
def stochastic(df, n=14, m=5, t=5):
    #데이터 프레임으로 받아오기 때문에 불필요

    #n 일중 최저가
    ndays_high = df['High'].rolling(window=n, min_periods=n).max()
    ndays_low = df['Low'].rolling(window=n, min_periods=n).min()
    fast_k = ((df['Close'] - ndays_low) / (ndays_high - ndays_low) * 100)
    slow_k = fast_k.ewm(span=m, min_periods=m).mean()
    slow_d = slow_k.ewm(span=t, min_periods=t).mean()
    df = df.assign(fast_k=fast_k, fast_d=slow_k, slow_k=slow_k, slow_d=slow_d)
    return df

# ===== raw data loading =====
# Get Stock List
list_name = 'Stock_List.csv'
sample_name = 'sample_submission_week4.csv'

# 종목 코드 로드
stock_list = read_csv("projects/asiae_project3/dacon_stockprediction/open_week4/Stock_List")
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))

# Get Data & Modeling
# 분석할 date 변수 지정
start_date = '20210104'
end_date = '20211029'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

# 모든 종목
stock_list.set_index("종목명", inplace=True)
selected_codes = stock_list.index.tolist()
stock_list = stock_list.loc[selected_codes]["종목코드"]

stock_dic = dict.fromkeys(selected_codes)
error_list = []
corr_list = []
anova_weekday = 0
anova_weeknum = 0
anova_mfi_signal = 0
timeunit_gap_forviz = 1
metric_days = 14
cat_vars = []
bin_vars = []
cat_vars.append("weekday")
cat_vars.append("weeknum")
bin_vars.append("mfi_signal")

# ==== selected feature =====
# selected_features = ["close", "kospi", "obv", "trading_amount_mv20", "mfi_signal"]
selected_features = None
logtrans_vec = ["close", "kospi", "volume", "trading_amount"]
pvalue_check = series(0, index=selected_features)

for stock_name, stock_code in stock_list.items():
    print("=====", stock_name, "=====")

    # 종목 주가 데이터 로드
    try:
        stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
        stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
        investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
        kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()
        # sleep(0.2)
        stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        investor_df.columns = ["Date", "inst", "fore"]
        kospi_df.columns = ["Date", "kospi"]
        # 영업일과 주가 정보를 outer 조인
        train_x = pd.merge(business_days, stock_df, how='left', on="Date")
        train_x = pd.merge(train_x, investor_df, how='left', on="Date")
        train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
        # 앞의 일자로 nan값 forward fill
        train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
        # 첫 날이 na 일 가능성이 있으므로 backward fill 수행
        train_x.iloc[:, 1:] = train_x.iloc[:, 1:].bfill(axis=0)
    except:
        stock_dic[stock_name] = dict.fromkeys(["df", "target_list"])
        stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
        # sleep(0.2)
        stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        # 영업일과 주가 정보를 outer 조인
        train_x = pd.merge(business_days, stock_df, how='left', on="Date")
        # 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
        # 앞의 일자로 nan값 forward fill
        train_x.iloc[:, 1:] = train_x.iloc[:, 1:].ffill(axis=0)
        # 첫 날이 na 일 가능성이 있으므로 backward fill 수행
        train_x.iloc[:, 1:] = train_x.iloc[:, 1:].bfill(axis=0)

    # create target list
    target_list = []
    target_list.append(train_x["Close"])
    target_list.append(train_x["Close"].shift(-1))
    target_list.append(train_x["Close"].shift(-2))
    target_list.append(train_x["Close"].shift(-3))
    target_list.append(train_x["Close"].shift(-4))
    target_list.append(train_x["Close"].shift(-5))
    for idx, value in enumerate(target_list):
        value.name = "target_shift" + str(idx)

    stock_dic[stock_name]["df"] = train_x.copy()
    stock_dic[stock_name]["target_list"] = target_list.copy()


for stock_name, stock_code in stock_dic.items():
    # ===== feature engineering =====
    # 요일 및 주차 파생변수 추가
    train_x['weekday'] = train_x["Date"].apply(lambda x: x.weekday())
    train_x['weeknum'] = train_x["Date"].apply(lambda x: week_of_month(x))

    # 거래대금 파생변수 추가
    train_x['trading_amount'] = train_x["Close"] * train_x["Volume"]

    # 월별 주기성 특징을 잡기 위한 sin 및 cos 변환 파생변수 추가
    day_to_sec = 24 * 60 * 60
    month_to_sec = 20 * day_to_sec
    timestamp_s = train_x["Date"].apply(datetime.timestamp)
    timestamp_freq = round((timestamp_s / month_to_sec).diff(20)[20], 1)

    train_x['dayofmonth_freq_sin'] = np.sin((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))
    train_x['dayofmonth_freq_cos'] = np.cos((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))


    # 1. OBV 파생변수 추가
    # 매수 신호: obv > obv_ema
    # 매도 신호: obv < obv_ema
    obv = [0]
    for i in range(1, len(train_x.Close)):
        if train_x.Close[i] >= train_x.Close[i - 1]:
            obv.append(obv[-1] + train_x.Volume[i])
        elif train_x.Close[i] < train_x.Close[i - 1]:
            obv.append(obv[-1] - train_x.Volume[i])
    train_x['obv'] = obv
    train_x['obv'][0] = nan
    train_x['obv_ema'] = train_x['obv'].ewm(com=metric_days, min_periods=metric_days).mean()

    # Stochastic 파생변수 추가
    # fast_d = moving average on fast_k
    train_x[["fast_k", "fast_d"]] = stochastic(train_x, n=metric_days)[["fast_k", "fast_d"]]

    # 2. MFI 파생변수 추가
    # MFI = 100 - (100 / 1 + MFR)
    # MFR = 14일간의 양의 MF / 14일간의 음의 MF
    # MF = 거래량 * (당일고가 + 당일저가 + 당일종가) / 3
    # MF 컬럼 만들기
    train_x["mf"] = train_x["Volume"] * ((train_x["High"]+train_x["Low"]+train_x["Close"]) / 3)
    # 양의 MF와 음의 MF 표기 컬럼 만들기
    p_n = []
    for i in range(len(train_x['mf'])):
        if i == 0 :
            p_n.append(nan)
        else:
            if train_x['mf'][i] >= train_x['mf'][i-1]:
                p_n.append('p')
            else:
                p_n.append('n')
    train_x['p_n'] = p_n
    # 14일간 양의 MF/ 14일간 음의 MF 계산하여 컬럼 만들기
    mfr = []
    for i in range(len(train_x['mf'])):
        if i < metric_days-1:
            mfr.append(nan)
        else:
            train_x_=train_x.iloc[(i-metric_days+1):i]
            a = (sum(train_x_['mf'][train_x['p_n'] == 'p']) + 1) / (sum(train_x_['mf'][train_x['p_n'] == 'n']) + 10)
            mfr.append(a)
    train_x['mfr'] = mfr
    # 최종 MFI 컬럼 만들기
    train_x['mfi'] = 100 - (100 / (1 + train_x['mfr']))
    train_x["mfi_signal"] = train_x['mfi'].apply(lambda x: "buy" if x > 50 else "sell")

    # 이동평균 추가
    train_x["close_mv5"] = train_x["Close"].rolling(5, min_periods=5).mean()
    train_x["close_mv10"] = train_x["Close"].rolling(10, min_periods=10).mean()
    train_x["close_mv20"] = train_x["Close"].rolling(20, min_periods=20).mean()

    train_x["volume_mv5"] = train_x["Volume"].rolling(5, min_periods=5).mean()
    train_x["volume_mv10"] = train_x["Volume"].rolling(10, min_periods=10).mean()
    train_x["volume_mv20"] = train_x["Volume"].rolling(20, min_periods=20).mean()

    train_x["trading_amount_mv5"] = train_x["trading_amount"].rolling(5, min_periods=5).mean()
    train_x["trading_amount_mv10"] = train_x["trading_amount"].rolling(10, min_periods=10).mean()
    train_x["trading_amount_mv20"] = train_x["trading_amount"].rolling(20, min_periods=20).mean()

    try:
        train_x["inst_mv5"] = train_x["inst"].rolling(5, min_periods=5).mean()
        train_x["inst_mv10"] = train_x["inst"].rolling(10, min_periods=10).mean()
        train_x["inst_mv20"] = train_x["inst"].rolling(20, min_periods=20).mean()

        train_x["fore_mv5"] = train_x["fore"].rolling(5, min_periods=5).mean()
        train_x["fore_mv10"] = train_x["fore"].rolling(10, min_periods=10).mean()
        train_x["fore_mv20"] = train_x["fore"].rolling(20, min_periods=20).mean()

        train_x["kospi_mv5"] = train_x["kospi"].rolling(5, min_periods=5).mean()
        train_x["kospi_mv10"] = train_x["kospi"].rolling(10, min_periods=10).mean()
        train_x["kospi_mv20"] = train_x["kospi"].rolling(20, min_periods=20).mean()
    except:
        pass

    # 지표계산을 위해 쓰인 컬럼 drop
    train_x.drop(["mf", "p_n", "mfr", "Open", "High", "Low"], inplace=True, axis=1)

    # 2021/1/4 이후 일자만 선택
    train_x = train_x[train_x["Date"] >= datetime(2021, 1, 4)]
    train_x = train_x.dropna()
    train_x.reset_index(drop=True, inplace=True)


    # 컬럼이름 소문자 변환 및 정렬
    train_x.columns = train_x.columns.str.lower()
    train_x = pd.concat([train_x[["date"]], train_x.iloc[:,1:].sort_index(axis=1)], axis=1)

    # <visualization>
    # 시각화용 데이터프레임 생성
    train_bi = pd.concat([target_list[timeunit_gap_forviz], train_x], axis=1)[:-timeunit_gap_forviz]

    # 기업 평균 상관관계를 측정하기 위한 연산
    corr_obj = train_bi.corr().round(3)
    corr_rows = corr_obj.index.tolist()
    corr_cols = corr_obj.columns.tolist()
    corr_list.append(corr_obj.to_numpy().round(3)[..., np.newaxis])

    # # 상관관계 시각화
    # fig, ax = plt.subplots(figsize=(16, 9))
    # graph = sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize": 8, "fontweight": "bold"})
    # plt.xticks(rotation=45)
    # fig.subplots_adjust(left=0.15, bottom=0.2)
    # ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
    # ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
    # plt.subplots_adjust(bottom=0.17, right=1)
    # plt.title('Correlation Visualization on ' + stock_name, fontsize=15, fontweight="bold", pad=15)
    # createFolder("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/")
    # plt.savefig("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/" + stock_name + ".png", dpi=300)
    # plt.close()

    # # feature 와 target 간 시각화
    # # ===== scatter plot on numerical feature =====
    # for i in train_x.columns:
    #     if i == "date":
    #         pass
    #     elif i in cat_vars + bin_vars:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         graph = sns.boxplot(x=train_bi[i], y=train_bi["target_shift" + str(timeunit_gap_forviz)], palette=sns.hls_palette())
    #         change_width(ax, 0.2)
    #         graph.set_title(i + " on " + stock_name, fontsize=15, fontweight="bold", pad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name + "/" + i + ".png", dpi=300)
    #         plt.close()
    #     else:
    #         fig, ax = plt.subplots(figsize=(12, 6))
    #         graph = sns.regplot(x=train_bi[i], y=train_bi["target_shift" + str(timeunit_gap_forviz)], color="green",
    #                             scatter_kws={'s': 15}, line_kws={"color": "orange"})
    #         graph.set_title(i + " on " + stock_name, fontsize=15, fontweight="bold", pad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/timegap_' + str(timeunit_gap_forviz) + "/" + stock_name + "/" + i +".png", dpi=300)
    #         plt.close()

    # # categorical, binary 변수에 대한 분산분석 (target 과의 상관관계 파악)
    # # 귀무가설(H0) : 두 변수는 상관관계가 없다
    # # 대립가설(H1) : 두 변수는 상관관계가 있다
    # pvalue_check_cat = train_bi.groupby("weekday")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    # anova_weekday += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0
    # pvalue_check_cat = train_bi.groupby("weeknum")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    # anova_weeknum += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0
    # pvalue_check_cat = train_bi.groupby("mfi_signal")["target_shift" + str(timeunit_gap_forviz)].apply(list)
    # anova_mfi_signal += 1 / len(stock_list) if f_oneway(*pvalue_check_cat)[1] <= 0.05 else 0

    # <feature selection>

    if selected_features is not None:
        tmp_list = [i for i in selected_features if i in train_x.columns]
        if len(tmp_list) != 0:
            train_x = train_x[tmp_list]

    # feature 분포 시각화
    # ===== hist plot on numerical feature =====
    # for i in train_x.columns:
    #     if i == "date" or i in cat_vars + bin_vars:
    #         pass
    #     else:
    #         plt.figure(figsize=(12, 6))
    #         graph = sns.histplot(x=train_bi[i], bins=50, color="orange")
    #         graph.set_title("Distribution on " + stock_name + " (skewness : " + str(train_bi[i].skew().round(3)) + ")", fontsize=15, fontweight="bold", pad=15)
    #         graph.set_xlabel(graph.get_xlabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         graph.set_ylabel(graph.get_ylabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name + "/dist_" + i + ".png", dpi=300)
    #         plt.close()

    # <feature scaling>
    # log transform
    for i in logtrans_vec:
        if i in train_x.columns:
            train_x[i] = train_x[i].apply(np.log1p)

    # transformation 후 재 시각화
    # ===== hist plot on numerical feature =====
    # for i in logtrans_vec:
    #     if i in train_x.columns:
    #         plt.figure(figsize=(12, 6))
    #         graph = sns.histplot(x=train_x[i], bins=50, color="orange")
    #         graph.set_title("After log scaling distribution on " + stock_name + " (skewness : " + str(train_x[i].skew().round(3)) + ")", fontsize=15,
    #                         fontweight="bold", pad=15)
    #         graph.set_xlabel(graph.get_xlabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         graph.set_ylabel(graph.get_ylabel(), fontsize=12, fontweight="bold", labelpad=15)
    #         plt.show()
    #         createFolder('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name)
    #         plt.savefig('projects/dacon_stockprediction/graphs/' + feature_test_seed + "/" + stock_name + "/dist_logTrans_" + i + ".png", dpi=300)
    #         plt.close()

    # # export csv for BI tool
    # corr_obj.to_csv("projects/dacon_stockprediction/bi_dataset/bi_corr_" + stock_name + ".csv", encoding="euc-kr", index_label=True, header=False)
    # train_bi.to_csv("projects/dacon_stockprediction/bi_dataset/bi_data_" + stock_name + ".csv", encoding="euc-kr", index=False)

    # onehot encoding
    onehot_encoder = MyOneHotEncoder()
    train_x = onehot_encoder.fit_transform(train_x, cat_vars + bin_vars)

# # 선택된 feature 들의 유의성 파악
# tmp_lm = sm.OLS(target_list[timeunit_gap_forviz][:-timeunit_gap_forviz].to_frame(),
#                 sm.add_constant(train_x, prepend=False)[:-timeunit_gap_forviz])
# ols_obj = tmp_lm.fit()
# pvalue_check += series([1 if i <= 0.05 else 0 for i in ols_obj.pvalues.drop("const")], index=ols_obj.pvalues.drop("const").index) / len(stock_list)



# # print(anova_weekday)
# # print(anova_weeknum)
# # print(anova_mfi_signal)
# pvalue_check.to_csv("projects/dacon_stockprediction/bi_dataset/pvalue_check_" + feature_test_seed + ".csv")
# easyIO(stock_dic, "projects/dacon_stockprediction/dataset/stock_dic_" + feature_test_seed + ".pickle", op="w")
# stock_dic = easyIO(None, "projects/dacon_stockprediction/dataset/stock_dic_" + feature_test_seed + ".pickle", op="r")

# data check
print(stock_dic["삼성전자"]["df"])

# # # 상관관계 평균 시각화
# corr_mean = dataframe(np.concatenate(corr_list, axis=2).mean(axis=2), index=corr_rows, columns=corr_cols).round(3)
# corr_std = dataframe(np.concatenate(corr_list, axis=2).std(axis=2), index=corr_rows, columns=corr_cols).round(3)
# print(corr_mean)
# print(corr_std)
#
# fig, ax = plt.subplots(figsize=(16, 9))
# graph = sns.heatmap(corr_mean, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Mean Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# corr_mean.to_csv("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/상관계수 평균csv.csv")
# plt.savefig("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/상관계수 평균.png", dpi=300)
# plt.close()
#
# fig, ax = plt.subplots(figsize=(16, 9))
# sns.heatmap(corr_std, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Std. Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# corr_std.to_csv("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/상관계수 표준편차csv.csv")
# plt.savefig("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/상관계수 표준편차.png", dpi=300)
# plt.close()
#
# fig, ax = plt.subplots(figsize=(16, 9))
# mean_to_std = corr_mean/corr_std
# sns.heatmap(mean_to_std, cmap="YlGnBu", linewidths=0.2, annot=True, annot_kws={"fontsize":8, "fontweight": "bold"})
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)
# ax.set_xticklabels(graph.get_xticklabels(), fontsize=8)
# ax.set_yticklabels(graph.get_yticklabels(), fontsize=8)
# plt.title('Mean divided by Std. Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# plt.subplots_adjust(bottom=0.17, right=1)
# mean_to_std.to_csv("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/표준편차 대비 상관계수csv.csv")
# plt.savefig("projects/dacon_stockprediction/graphs/timegap_" + str(timeunit_gap_forviz) + "/표준편차 대비 상관계수.png", dpi=300)
# plt.close()

# print(anova_weekday)
# print(anova_weeknum)
# print(error_list)








# ===== train, val, test split and auto prediction  =====
# train 2021/1/6 ~ 2021/9/5
# validation 2021/9/6 ~ 2021/9/17
# test 2021/9/27 ~ 2021/10/1

# 학습 전 필요 변수 초기화
kfolds_spliter = TimeSeriesSplit(n_splits=4, test_size=5, gap=5)
targetType = "numeric"
targetTask = None
class_levels = [0,1]
cut_off = 0

ds = None
result_val = None
result_test = None


# --- Modeling ---
def doLinear(train_x, train_y, test_x=None, test_y=None, model_export=False, preTrained=None):
    result_dic = {}
    scaler_standard = prep.StandardScaler()
    train_x = scaler_standard.fit_transform(train_x)
    test_x = scaler_standard.transform(test_x)

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            result_dic["model"] = lm.LinearRegression(n_jobs=multiprocessing.cpu_count())

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:

                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = lm.LogisticRegression(max_iter=1000, penalty="none", multi_class="ovr", random_state=3)

            result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = lm.LogisticRegression(max_iter=1000, penalty="none", multi_class="multinomial",
                                                            n_jobs=multiprocessing.cpu_count(), random_state=3)

            result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                # axis=0 ---> column, axis=1 ---> row
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic


# result_val["Linear"] = doLinear(ds["train_x_oh"], ds["train_y"],
#                                 ds["val_x_oh"], ds["val_y"],
#                                 model_export=True)
# print(result_val["Linear"]["performance"])

# result_test["Linear"] = doLinear(ds["train_x_oh"], ds["train_y"],
#                                  ds["test_x_oh"], None,
#                                  preTrained=result_val["Linear"]["model"])
# print(result_val["Linear"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

def doElasticNet(train_x, train_y, test_x=None, test_y=None, tuningMode=True,
                 c=None, alpha=None, l1_ratio=None,
                 kfolds=KFold(10, shuffle=True, random_state=2323),
                 model_export=False, preTrained=None, seed=1000):
    result_dic = {}
    scaler_standard = prep.StandardScaler()
    train_x = scaler_standard.fit_transform(train_x)

    runStart = time()
    if targetType == "numeric":
        tuner_params = {"alpha": np.linspace(1e-3, 1e+3, 100).tolist(),
                        "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99]}
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                model_tuner = GridTuner(lm.ElasticNet(max_iter=1000, random_state=seed),
                                        refit=False,
                                        param_grid=tuner_params,
                                        scoring="neg_root_mean_squared_error",
                                        cv=kfolds.split(train_x, train_y),
                                        n_jobs=multiprocessing.cpu_count())
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = lm.ElasticNet(max_iter=1000, alpha=model_tuner.best_params_["alpha"],
                                                    l1_ratio=model_tuner.best_params_["l1_ratio"],
                                                    random_state=seed)
            else:
                result_dic["model"] = lm.ElasticNet(max_iter=1000,
                                                    alpha=alpha,
                                                    l1_ratio=l1_ratio,
                                                    random_state=seed)

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_standard.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        tuner_params = {"C": np.linspace(1e-3, 1e+3, 100).tolist(),
                        "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99]}

        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:
                    model_tuner = GridTuner(
                        lm.LogisticRegression(max_iter=1000, penalty="elasticnet", solver="saga", multi_class="ovr", random_state=seed),
                        refit=False,
                        param_grid=tuner_params,
                        scoring="neg_log_loss",
                        cv=kfolds.split(train_x, train_y),
                        n_jobs=multiprocessing.cpu_count())
                    model_tuner.fit(train_x, train_y)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lm.LogisticRegression(penalty="elasticnet", solver="saga", multi_class="ovr",
                                                                random_state=seed, n_jobs=multiprocessing.cpu_count(),
                                                                max_iter=1000, C=c,
                                                                l1_ratio=model_tuner.best_params_["l1_ratio"])
                else:
                    result_dic["model"] = lm.LogisticRegression(penalty="elasticnet", solver="saga", multi_class="ovr",
                                                                random_state=seed, n_jobs=multiprocessing.cpu_count(),
                                                                max_iter=1000, C=c, l1_ratio=l1_ratio)

            result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_standard.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:

                    model_tuner = GridTuner(
                        lm.LogisticRegression(max_iter=1000, penalty="elasticnet", solver="saga", multi_class="multinomial", random_state=seed),
                        refit=False,
                        param_grid=tuner_params,
                        scoring="neg_log_loss",
                        cv=kfolds.split(train_x, train_y),
                        n_jobs=multiprocessing.cpu_count())
                    model_tuner.fit(train_x, train_y)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lm.LogisticRegression(penalty="elasticnet", solver="saga", multi_class="multinomial",
                                                                random_state=seed, n_jobs=multiprocessing.cpu_count(),
                                                                max_iter=1000,
                                                                C=model_tuner.best_params_["C"],
                                                                l1_ratio=model_tuner.best_params_["l1_ratio"])
                else:
                    result_dic["model"] = lm.LogisticRegression(penalty="elasticnet", solver="saga", multi_class="multinomial",
                                                                random_state=seed, n_jobs=multiprocessing.cpu_count(),
                                                                max_iter=1000,
                                                                C=c, l1_ratio=l1_ratio)

            result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_standard.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                # axis=0 ---> column, axis=1 ---> row
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# result_val["ElasticNet"] = doElasticNet(ds["train_x_oh"], ds["train_y"],
#                                         ds["val_x_oh"], ds["val_y"],
#                                         kfolds=kfolds_spliter,
#                                         model_export=True)
# print(result_val["ElasticNet"]["performance"])
# print(result_val["ElasticNet"]["running_time"])
#
# result_test["ElasticNet"] = doElasticNet(ds["full_x_oh"], ds["full_y"],
#                                          ds["test_x_oh"], None,
#                                          preTrained=result_val["ElasticNet"]["model"])
# print(result_val["ElasticNet"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

# result_test["ElasticNet"] = doElasticNet(ds["train_x_oh"], ds["train_y"],
#                                          ds["test_x_oh"], None,
#                                          preTrained=result_val["ElasticNet"]["model"])
# print(result_val["ElasticNet"]["pred"][:10])


# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


# classification only
def doQDA(train_x, train_y, test_x=None, test_y=None, model_export=False, preTrained=None):
    result_dic = {}
    scaler_standard = prep.StandardScaler()
    train_x = scaler_standard.fit_transform(train_x)

    runStart = time()
    if targetType == "numeric":
        print("Only class type")
        return None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = qda()
            result_dic["model"] = qda().fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_standard.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = qda()
            result_dic["model"] = qda().fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_standard.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                # axis=0 3---> column, axis=1 ---> row
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# result_val["QDA"] = doQDA(ds["train_x_oh"], ds["train_y"],
#                           ds["val_x_oh"], ds["val_y"],
#                           model_export=True)
# print(result_val["QDA"]["performance"])
# print(result_val["QDA"]["running_time"])
#
# result_test["QDA"] = doQDA(ds["full_x_oh"], ds["full_y"],
#                            ds["test_x_oh"], None,
#                            preTrained=result_val["QDA"]["model"])
# print(result_val["QDA"]["pred"][:10])



def doSVM(train_x, train_y, test_x=None, test_y=None,
          kernelSeq=["linear", "poly", "rbf"],
          costSeq=[pow(10, i) for i in [-2, -1, 0, 1, 2]],
          gammaSeq=[pow(10, i) for i in [-2, -1, 0, 1, 2]],
          kfolds=KFold(10, shuffle=True, random_state=2323),
          model_export=False, preTrained=None, seed=7777):
    result_dic = {}
    scaler_minmax = MinMaxScaler()
    train_x = scaler_minmax.fit_transform(train_x)
    np.random.seed(seed)
    tuner_params = {"kernel": kernelSeq,
                    "C": costSeq,
                    "gamma": gammaSeq}

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            model_tuner = GridTuner(svm.SVR(max_iter=1000), param_grid=tuner_params,
                                    cv=kfolds.split(train_x, train_y), refit=False,
                                    n_jobs=multiprocessing.cpu_count(),
                                    pre_dispatch=multiprocessing.cpu_count(),
                                    scoring="neg_root_mean_squared_error")
            model_tuner.fit(train_x, train_y)

            print("Tuning Result --->", model_tuner.best_params_)
            result_dic["best_params"] = model_tuner.best_params_

            result_dic["model"] = svm.SVR(max_iter=1000, kernel=model_tuner.best_params_["kernel"],
                                          C=model_tuner.best_params_["C"],
                                          gamma=model_tuner.best_params_["gamma"])
        result_dic["model"].fit(train_x, train_y)

        if test_x is not None:
            test_x = scaler_minmax.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                model_tuner = GridTuner(svm.SVC(max_iter=1000, probability=True, random_state=seed),
                                        param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        pre_dispatch=multiprocessing.cpu_count(),
                                        scoring="neg_log_loss")
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = svm.SVC(max_iter=1000, probability=True,
                                              kernel=model_tuner.best_params_["kernel"],
                                              C=model_tuner.best_params_["C"],
                                              gamma=model_tuner.best_params_["gamma"])
            result_dic["model"].fit(train_x, train_y)

            if test_x is not None:
                test_x = scaler_minmax.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                model_tuner = GridTuner(svm.SVC(max_iter=1000, probability=True, random_state=seed),
                                        param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        pre_dispatch=multiprocessing.cpu_count(),
                                        scoring="neg_log_loss")
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = svm.SVC(max_iter=1000, probability=True,
                                              kernel=model_tuner.best_params_["kernel"],
                                              C=model_tuner.best_params_["C"],
                                              gamma=model_tuner.best_params_["gamma"])
            result_dic["model"].fit(train_x, train_y)

            if test_x is not None:
                test_x = scaler_minmax.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# result_val["SVM"] = doSVM(ds["train_x_oh"], ds["train_y"],
#                           ds["val_x_oh"], ds["val_y"],
#                           kfolds=kfolds_spliter,
#                           model_export=True)
# print(result_val["SVM"]["best_params"])
# print(result_val["SVM"]["performance"])
# print(result_val["SVM"]["running_time"])
#
# result_test["SVM"] = doSVM(ds["full_x_oh"], ds["full_y"],
#                            ds["test_x_oh"], None,
#                            preTrained=result_val["SVM"]["model"])
# print(result_test["SVM"]["pred"][:10])
#
# # # save obejcts
# # easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# # easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")
#
#
# # display the performance
# for k, v in result_val.items():
#     if v is not None:
#         if "performance" in v.keys():
#             print(k, "--->", v["performance"])


def doKNN(train_x, train_y, test_x=None, test_y=None, kSeq=[3, 5, 7],
          kfolds=KFold(10, shuffle=True, random_state=2323),
          model_export=False, preTrained=None, seed=7777):
    result_dic = {}
    np.random.seed(seed)
    scaler_minmax = MinMaxScaler()
    train_x = scaler_minmax.fit_transform(train_x)
    tuner_params = {"n_neighbors": kSeq}

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            knn_model = knn.KNeighborsRegressor(n_jobs=None)
            model_tuner = GridTuner(knn_model, param_grid=tuner_params,
                                    cv=kfolds.split(train_x, train_y), refit=False,
                                    n_jobs=multiprocessing.cpu_count(),
                                    pre_dispatch=multiprocessing.cpu_count(),
                                    scoring="neg_root_mean_squared_error")
            model_tuner.fit(train_x, train_y)

            print("Tuning Result --->", model_tuner.best_params_)
            result_dic["best_params"] = model_tuner.best_params_

            result_dic["model"] = knn.KNeighborsRegressor(n_neighbors=model_tuner.best_params_["n_neighbors"],
                                                          n_jobs=multiprocessing.cpu_count())

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_minmax.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                knn_model = knn.KNeighborsClassifier(n_jobs=None)
                model_tuner = GridTuner(knn_model, param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        pre_dispatch=multiprocessing.cpu_count(),
                                        scoring="neg_log_loss")
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = knn.KNeighborsClassifier(n_neighbors=model_tuner.best_params_["n_neighbors"],
                                                               n_jobs=multiprocessing.cpu_count())

            result_dic["model"] = result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_minmax.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                knn_model = knn.KNeighborsClassifier(n_jobs=None)
                model_tuner = GridTuner(knn_model, param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        pre_dispatch=multiprocessing.cpu_count(),
                                        scoring="neg_log_loss")
                model_tuner.fit(train_x, train_y)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = knn.KNeighborsClassifier(n_neighbors=model_tuner.best_params_["n_neighbors"],
                                                               n_jobs=multiprocessing.cpu_count())

            result_dic["model"].fit(train_x, train_y)
            if test_x is not None:
                test_x = scaler_minmax.transform(test_x)
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic


# kSeq = list(range(3, int(ds["train_x"].shape[0] * 0.05), 2))

# result_val["KNN"] = doKNN(ds["train_x_oh"], ds["train_y"],
#                           ds["val_x_oh"], ds["val_y"],
#                           kfolds=kfolds_spliter,
#                           kSeq=kSeq, model_export=True)
# print(result_val["KNN"]["best_params"])
# print(result_val["KNN"]["performance"])
# print(result_val["KNN"]["running_time"])
#
# result_test["KNN"] = doKNN(ds["full_x"], ds["full_y"],
#                            ds["test_x"], None,
#                            preTrained=result_val["KNN"]["model"])
# print(result_test["KNN"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

# for k, v in result_val.items():
#     if v is not None:
#         if "performance" in v.keys():
#             print(k, "--->", v["performance"])

def doXGB(train_x, train_y, test_x=None, test_y=None, ntrees=5000, eta=5e-3,
          depthSeq=[4, 6, 8], subsampleSeq=[0.6, 0.8], colsampleSeq=[0.6, 0.8],
          l2Seq=[0.1, 1.0, 5.0], mcwSeq=[1, 3, 5], gammaSeq=[0.0, 0.2],
          kfolds=KFold(10, shuffle=True, random_state=2323),
          model_export=False, preTrained=None, seed=11, tuningMode=True):
    result_dic = {}
    np.random.seed(seed)
    tuner_params = {"max_depth": depthSeq, "subsample": subsampleSeq, "colsample_bytree": colsampleSeq,
                    "reg_lambda": l2Seq, "min_child_weight": mcwSeq, "gamma": gammaSeq}
    patientRate = 0.2

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                xgb_model = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                             n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                             n_jobs=None, random_state=seed,
                                             verbosity=0, use_label_encoder=False)
                model_tuner = GridTuner(xgb_model, param_grid=tuner_params, cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        scoring="neg_root_mean_squared_error")
                model_tuner.fit(train_x, train_y, verbose=False)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=ntrees, learning_rate=eta,
                                                       max_depth=model_tuner.best_params_["max_depth"],
                                                       subsample=model_tuner.best_params_["subsample"],
                                                       colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                       reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                       min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                       gamma=model_tuner.best_params_["gamma"],
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)

                result_dic["best_params"]["best_trees"] = 0
                for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                    result_dic["model"].fit(train_x.iloc[nonkIdx,:], train_y[nonkIdx],
                                            eval_set=[(train_x.iloc[kIdx,:], train_y[kIdx])],
                                            eval_metric="rmse", verbose=False,
                                            early_stopping_rounds=int(ntrees * patientRate))
                    result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration / kfolds.get_n_splits()
                result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                       max_depth=model_tuner.best_params_["max_depth"],
                                                       subsample=model_tuner.best_params_["subsample"],
                                                       colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                       reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                       min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                       gamma=model_tuner.best_params_["gamma"],
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)
            else:
                result_dic["model"] = xgb.XGBRegressor(booster="gbtree", objective="reg:squarederror",
                                                       n_estimators=ntrees, learning_rate=eta,
                                                       max_depth=depthSeq,
                                                       subsample=subsampleSeq,
                                                       colsample_bytree=colsampleSeq,
                                                       reg_lambda=l2Seq,
                                                       min_child_weight=mcwSeq,
                                                       gamma=gammaSeq,
                                                       n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                       verbosity=0, use_label_encoder=False)

        result_dic["model"].fit(train_x, train_y, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:


                    xgb_model = xgb.XGBClassifier(booster="gbtree", objective="binary:logistic",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  n_jobs=None, random_state=seed,
                                                  verbosity=0, use_label_encoder=False)
                    model_tuner = GridTuner(xgb_model, param_grid=tuner_params, cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_log_loss")
                    model_tuner.fit(train_x, train_y, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="binary:logistic",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx],
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])],
                                                eval_metric="logloss", verbose=False,
                                                early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="binary:logistic",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)

                else:
                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="binary:logistic",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=depthSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            gamma=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)

            result_dic["model"].fit(train_x, train_y, verbose=False)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:


                    xgb_model = xgb.XGBClassifier(booster="gbtree", objective="multi:softmax",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  n_jobs=None, random_state=seed,
                                                  verbosity=0, use_label_encoder=False)
                    model_tuner = GridTuner(xgb_model, param_grid=tuner_params, cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_log_loss")
                    model_tuner.fit(train_x, train_y, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="multi:softmax",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx],
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])],
                                                eval_metric="mlogloss", verbose=False,
                                                early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="multi:softmax",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)
                else:
                    result_dic["model"] = xgb.XGBClassifier(booster="gbtree", objective="multi:softmax",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=depthSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            gamma=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            verbosity=0, use_label_encoder=False)

            result_dic["model"].fit(train_x, train_y, verbose=False)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# # xgboost - gbtree
# result_val["XGB_GBT"] = doXGB(ds["train_x_oh"], ds["train_y"],
#                               ds["val_x_oh"], ds["val_y"],
#                               kfolds=kfolds_spliter,
#                               ntrees=5000,
#                               model_export=True)
# print(result_val["XGB_GBT"]["best_params"])
# print(result_val["XGB_GBT"]["performance"])
# print(result_val["XGB_GBT"]["running_time"])
#
# # xgboost - gbtree
# result_test["XGB_GBT"] = doXGB(ds["full_x_oh"], ds["full_y"],
#                                ds["test_x_oh"], None,
#                                preTrained=result_val["XGB_GBT"]["model"])
# print(result_test["XGB_GBT"]["pred"][:10])
#
# # # save obejcts
# # easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# # easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


def doLGB(train_x, train_y, test_x=None, test_y=None, categoIdx=None, boostingType="goss", ntrees=5000, eta=5e-3,
          leavesSeq=[pow(2, i) - 1 for i in [4, 6, 8]], subsampleSeq=[0.6, 0.8], gammaSeq=[0.0, 0.2],
          colsampleSeq=[0.6, 0.8], l2Seq=[0.1, 1.0, 5.0], mcsSeq=[5, 10, 20], mcwSeq=[1e-4, 1e-3, 1e-2],
          kfolds=KFold(10, shuffle=True, random_state=2323), model_export=False, preTrained=None, seed=22, tuningMode=True):
    result_dic = {}
    np.random.seed(seed)
    tuner_params = {"num_leaves": leavesSeq, "subsample": subsampleSeq, "colsample_bytree": colsampleSeq,
                    "reg_lambda": l2Seq, "min_child_samples": mcsSeq, "min_child_weight": mcwSeq, "min_split_gain": gammaSeq}
    patientRate = 0.2

    runStart = time()

    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:
                if boostingType == "rf":
                    lgb_model = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  silent=True, n_jobs=None,
                                                  subsample_freq=2, random_state=seed)
                    model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_root_mean_squared_error")
                    model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="rmse",
                                                verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)
                elif boostingType == "goss":
                    lgb_model = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                  n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                  silent=True, n_jobs=None, random_state=seed)
                    model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_root_mean_squared_error")
                    model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
                    result_dic["model"] = lgb_model.fit(train_x, train_y, categorical_feature=categoIdx,
                                                        eval_set=[(test_x, test_y)], eval_metric="rmse",
                                                        verbose=False, early_stopping_rounds=int(ntrees * patientRate))

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="rmse",
                                                verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            num_leaves=model_tuner.best_params_["num_leaves"],
                                                            subsample=model_tuner.best_params_["subsample"],
                                                            colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                            reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                            min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                            min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                            min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
            else: # not tuning mode
                if boostingType == "rf":
                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=leavesSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            min_child_samples=mcsSeq,
                                                            min_split_gain=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                            subsample_freq=2, silent=True)
                elif boostingType == "goss":
                    result_dic["model"] = lgb.LGBMRegressor(boosting_type=boostingType, objective="regression",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            num_leaves=leavesSeq,
                                                            subsample=subsampleSeq,
                                                            colsample_bytree=colsampleSeq,
                                                            reg_lambda=l2Seq,
                                                            min_child_weight=mcwSeq,
                                                            min_child_samples=mcsSeq,
                                                            min_split_gain=gammaSeq,
                                                            n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)


        result_dic["model"].fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:

                    if boostingType == "rf":
                        lgb_model = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                       n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                       silent=True, n_jobs=None,
                                                       subsample_freq=2, random_state=seed)

                        model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                                cv=kfolds.split(train_x, train_y), refit=False,
                                                n_jobs=multiprocessing.cpu_count(),
                                                scoring="neg_log_loss")
                        model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                        print("Tuning Result --->", model_tuner.best_params_)
                        result_dic["best_params"] = model_tuner.best_params_

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)

                        result_dic["best_params"]["best_trees"] = 0
                        for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                            result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                    eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="binary_logloss",
                                                    verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                            result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                        result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)
                    elif boostingType == "goss":
                        lgb_model = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                       n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                       silent=True, n_jobs=None,
                                                       random_state=seed)
                        model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                                cv=kfolds.split(train_x, train_y), refit=False,
                                                n_jobs=multiprocessing.cpu_count(),
                                                scoring="neg_log_loss")
                        model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                        print("Tuning Result --->", model_tuner.best_params_)
                        result_dic["best_params"] = model_tuner.best_params_

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)

                        result_dic["best_params"]["best_trees"] = 0
                        for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                            result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                    eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="binary_logloss",
                                                    verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                            result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                        result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
                else: # not tuning mode
                    if boostingType == "rf":
                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=leavesSeq,
                                                                 subsample=subsampleSeq,
                                                                 colsample_bytree=colsampleSeq,
                                                                 reg_lambda=l2Seq,
                                                                 min_child_weight=mcwSeq,
                                                                 min_child_samples=mcsSeq,
                                                                 min_split_gain=gammaSeq,
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)
                    elif boostingType == "goss":
                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="binary",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=leavesSeq,
                                                                 subsample=subsampleSeq,
                                                                 colsample_bytree=colsampleSeq,
                                                                 reg_lambda=l2Seq,
                                                                 min_child_weight=mcwSeq,
                                                                 min_child_samples=mcsSeq,
                                                                 min_split_gain=gammaSeq,
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)

            result_dic["model"].fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:

                    if boostingType == "rf":
                        lgb_model = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                       n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                       silent=True, n_jobs=None,
                                                       subsample_freq=2, random_state=seed)
                        model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                                cv=kfolds.split(train_x, train_y), refit=False,
                                                n_jobs=multiprocessing.cpu_count(),
                                                scoring="neg_log_loss")
                        model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                        print("Tuning Result --->", model_tuner.best_params_)
                        result_dic["best_params"] = model_tuner.best_params_

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)

                        result_dic["best_params"]["best_trees"] = 0
                        for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                            result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                    eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="multi_logloss",
                                                    verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                            result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                        result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)
                    elif boostingType == "goss":
                        lgb_model = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                       n_estimators=int(ntrees * patientRate), learning_rate=eta / 10,
                                                       colsample_bytree=0.8, silent=True,
                                                       n_jobs=None, random_state=seed)
                        model_tuner = GridTuner(lgb_model, param_grid=tuner_params,
                                                cv=kfolds.split(train_x, train_y), refit=False,
                                                n_jobs=multiprocessing.cpu_count(),
                                                scoring="neg_log_loss")
                        model_tuner.fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)

                        print("Tuning Result --->", model_tuner.best_params_)
                        result_dic["best_params"] = model_tuner.best_params_

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)

                        result_dic["best_params"]["best_trees"] = 0
                        for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                            result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], categorical_feature=categoIdx,
                                                    eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], eval_metric="multi_logloss",
                                                    verbose=False, early_stopping_rounds=int(ntrees * patientRate))
                            result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                        result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 num_leaves=model_tuner.best_params_["num_leaves"],
                                                                 subsample=model_tuner.best_params_["subsample"],
                                                                 colsample_bytree=model_tuner.best_params_["colsample_bytree"],
                                                                 reg_lambda=model_tuner.best_params_["reg_lambda"],
                                                                 min_child_weight=model_tuner.best_params_["min_child_weight"],
                                                                 min_child_samples=model_tuner.best_params_["min_child_samples"],
                                                                 min_split_gain=model_tuner.best_params_["min_split_gain"],
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)
                else:
                    if boostingType == "rf":
                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=leavesSeq,
                                                                 subsample=subsampleSeq,
                                                                 colsample_bytree=colsampleSeq,
                                                                 reg_lambda=l2Seq,
                                                                 min_child_weight=mcwSeq,
                                                                 min_child_samples=mcsSeq,
                                                                 min_split_gain=gammaSeq,
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed,
                                                                 subsample_freq=2, silent=True)
                    elif boostingType == "goss":
                        result_dic["model"] = lgb.LGBMClassifier(boosting_type=boostingType, objective="multiclass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 num_leaves=leavesSeq,
                                                                 subsample=subsampleSeq,
                                                                 colsample_bytree=colsampleSeq,
                                                                 reg_lambda=l2Seq,
                                                                 min_child_weight=mcwSeq,
                                                                 min_child_samples=mcsSeq,
                                                                 min_split_gain=gammaSeq,
                                                                 n_jobs=multiprocessing.cpu_count(), random_state=seed, silent=True)

            result_dic["model"].fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# # lightgbm - randomforest
# result_val["LGB_RF"] = doLGB(ds["train_x"], ds["train_y"],
#                              ds["val_x"], ds["val_y"],
#                              categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                              kfolds=kfolds_spliter,
#                              boostingType="rf", ntrees=500,
#                              model_export=True)
# print(result_val["LGB_RF"]["best_params"])
# print(result_val["LGB_RF"]["performance"])
# print(result_val["LGB_RF"]["running_time"])
#
# result_test["LGB_RF"] = doLGB(ds["full_x"], ds["full_y"],
#                               ds["test_x"], None,
#                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                               preTrained=result_val["LGB_RF"]["model"])
# print(result_test["LGB_RF"]["pred"][:10])

# lightgbm - goss
# result_val["LGB_GOSS"] = doLGB(ds["train_x"], ds["train_y"],
#                                ds["val_x"], ds["val_y"],
#                                categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                                kfolds=kfolds_spliter,
#                                boostingType="goss", ntrees=5000,
#                                model_export=True)
# print(result_val["LGB_GOSS"]["best_params"])
# print(result_val["LGB_GOSS"]["performance"])
# print(result_val["LGB_GOSS"]["running_time"])


# result_test["LGB_GOSS"] = doLGB(ds["full_x"], ds["full_y"],
#                                 ds["test_x"], None,
#                                 categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                                 preTrained=result_val["LGB_GOSS"]["model"])
# print(result_test["LGB_GOSS"]["pred"][:10])


# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


def doCAT(train_x, train_y, test_x=None, test_y=None, categoIdx=None, boostingType="Plain", ntrees=5000, eta=5e-2,
          depthSeq=[4, 6, 8], bagTempSeq=[0.2, 0.8], colsampleSeq=[0.6, 0.8], l2Seq=[0.1, 1.0, 5.0], random_strength=[0.1, 1.0],
          kfolds=KFold(10, shuffle=True, random_state=2323), model_export=False, preTrained=None, seed=33, tuningMode=True):
    result_dic = {}
    np.random.seed(seed)
    tuner_params = {"max_depth": depthSeq, "bagging_temperature": bagTempSeq, "rsm": colsampleSeq,
                    "l2_leaf_reg": l2Seq, "random_strength": random_strength}
    patientRate = 0.2

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            if tuningMode:

                cat_model = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="RMSE",
                                                  n_estimators=int(ntrees / 2), learning_rate=eta / 10,
                                                  logging_level="Silent", thread_count=None, random_state=seed)
                model_tuner = GridTuner(cat_model, param_grid=tuner_params,
                                        cv=kfolds.split(train_x, train_y), refit=False,
                                        n_jobs=multiprocessing.cpu_count(),
                                        scoring="neg_root_mean_squared_error")
                model_tuner.fit(train_x, train_y, cat_features=categoIdx)

                print("Tuning Result --->", model_tuner.best_params_)
                result_dic["best_params"] = model_tuner.best_params_

                result_dic["model"] = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="RMSE",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                            rsm=model_tuner.best_params_["rsm"],
                                                            l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                            random_strength=model_tuner.best_params_["random_strength"],
                                                            thread_count=multiprocessing.cpu_count(),
                                                            logging_level="Silent", random_state=seed)

                result_dic["best_params"]["best_trees"] = 0
                for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                    result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], cat_features=categoIdx,
                                            eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], use_best_model=True,
                                            early_stopping_rounds=int(ntrees * patientRate))
                    result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                result_dic["model"] = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="RMSE",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                            rsm=model_tuner.best_params_["rsm"],
                                                            l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                            random_strength=model_tuner.best_params_["random_strength"],
                                                            thread_count=multiprocessing.cpu_count(),
                                                            logging_level="Silent", random_state=seed)
            else:
                result_dic["model"] = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="RMSE",
                                                            n_estimators=ntrees, learning_rate=eta,
                                                            max_depth=depthSeq,
                                                            bagging_temperature=bagTempSeq,
                                                            rsm=colsampleSeq,
                                                            l2_leaf_reg=l2Seq,
                                                            random_strength=random_strength,
                                                            thread_count=multiprocessing.cpu_count(),
                                                            logging_level="Silent", random_state=seed)

        result_dic["model"].fit(train_x, train_y, cat_features=categoIdx)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:

                    cat_model = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="Logloss",
                                                       n_estimators=int(ntrees / 2), learning_rate=eta / 10,
                                                       logging_level="Silent", thread_count=None, random_state=seed)
                    model_tuner = GridTuner(cat_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_log_loss")
                    model_tuner.fit(train_x, train_y, cat_features=categoIdx)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="Logloss",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 max_depth=model_tuner.best_params_["max_depth"],
                                                                 rsm=model_tuner.best_params_["rsm"],
                                                                 l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                                 random_strength=model_tuner.best_params_["random_strength"],
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], cat_features=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], use_best_model=True,
                                                early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="Logloss",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 max_depth=model_tuner.best_params_["max_depth"],
                                                                 bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                                 rsm=model_tuner.best_params_["rsm"],
                                                                 l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                                 random_strength=model_tuner.best_params_["random_strength"],
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)
                else: # not tuning mode
                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="Logloss",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 max_depth=depthSeq,
                                                                 bagging_temperature=bagTempSeq,
                                                                 rsm=colsampleSeq,
                                                                 l2_leaf_reg=l2Seq,
                                                                 random_strength=random_strength,
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)

            result_dic["model"].fit(train_x, train_y, cat_features=categoIdx)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if tuningMode:

                    cat_model = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="MultiClass",
                                                       n_estimators=int(ntrees / 2), learning_rate=eta / 10,
                                                       logging_level="Silent", thread_count=None, random_state=seed)
                    model_tuner = GridTuner(cat_model, param_grid=tuner_params,
                                            cv=kfolds.split(train_x, train_y), refit=False,
                                            n_jobs=multiprocessing.cpu_count(),
                                            scoring="neg_log_loss")
                    model_tuner.fit(train_x, train_y, cat_features=categoIdx)

                    print("Tuning Result --->", model_tuner.best_params_)
                    result_dic["best_params"] = model_tuner.best_params_

                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="MultiClass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 max_depth=model_tuner.best_params_["max_depth"],
                                                                 rsm=model_tuner.best_params_["rsm"],
                                                                 l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                                 random_strength=model_tuner.best_params_["random_strength"],
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)

                    result_dic["best_params"]["best_trees"] = 0
                    for nonkIdx, kIdx in kfolds.split(train_x, train_y):
                        result_dic["model"].fit(train_x.iloc[nonkIdx, :], train_y[nonkIdx], cat_features=categoIdx,
                                                eval_set=[(train_x.iloc[kIdx, :], train_y[kIdx])], use_best_model=True,
                                                early_stopping_rounds=int(ntrees * patientRate))
                        result_dic["best_params"]["best_trees"] += result_dic["model"].best_iteration_ / kfolds.get_n_splits()
                    result_dic["best_params"]["best_trees"] = int(result_dic["best_params"]["best_trees"])

                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="MultiClass",
                                                                 n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                                 max_depth=model_tuner.best_params_["max_depth"],
                                                                 bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                                 rsm=model_tuner.best_params_["rsm"],
                                                                 l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                                 random_strength=model_tuner.best_params_["random_strength"],
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)
                else:
                    result_dic["model"] = cat.CatBoostClassifier(boosting_type=boostingType, loss_function="MultiClass",
                                                                 n_estimators=ntrees, learning_rate=eta,
                                                                 max_depth=depthSeq,
                                                                 bagging_temperature=bagTempSeq,
                                                                 rsm=colsampleSeq,
                                                                 l2_leaf_reg=l2Seq,
                                                                 random_strength=random_strength,
                                                                 thread_count=multiprocessing.cpu_count(),
                                                                 logging_level="Silent", random_state=seed)

            result_dic["model"].fit(train_x, train_y, cat_features=categoIdx)
            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# # CatBoost - GBM
# result_val["CAT_GBM"] = doCAT(ds["train_x"], ds["train_y"],
#                               ds["val_x"], ds["val_y"],
#                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                               kfolds=kfolds_spliter,
#                               boostingType="Plain", ntrees=5000,
#                               model_export=True)
#
# print(result_val["CAT_GBM"]["best_params"])
# print(result_val["CAT_GBM"]["performance"])
# print(result_val["CAT_GBM"]["running_time"])

# result_test["CAT_GBM"] = doCAT(ds["full_x"], ds["full_y"],
#                                ds["test_x"], None,
#                                categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                                preTrained=result_val["CAT_GBM"]["model"])
# print(result_test["CAT_GBM"]["pred"][:10])



# # CatBoost - Ordered Boosting
# result_val["CAT_ORD"] = doCAT(ds["train_x"], ds["train_y"],
#                               ds["val_x"], ds["val_y"],
#                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                               boostingType="Ordered", ntrees=5000,
#                               model_export=True)
#
# print(result_val["CAT_ORD"]["best_params"])
# print(result_val["CAT_ORD"]["performance"])
# print(result_val["CAT_ORD"]["running_time"])
#
# result_test["CAT_ORD"] = doCAT(ds["full_x"], ds["full_y"],
#                                ds["test_x"], None,
#                                categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
#                                preTrained=result_val["CAT_ORD"]["model"])
# print(result_test["CAT_ORD"]["pred"][:10])


# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


# for k, v in result_val.items():
#     if v is not None:
#         if "cv_score" in v.keys():
#             print("===", k, "===\n", v["cv_score"])





class MyHyperModel(HyperModel):
    def __init__(self, nCols, mlpName, hiddenLayers, dropoutRate, seqLength=None, eta=1e-3):
        self.nCols = nCols
        self.mlpName = mlpName
        self.hiddenLayers = hiddenLayers
        self.dropoutRate = dropoutRate
        self.seqLength = seqLength
        self.eta = eta
    def build(self, hp):
        self.hp_hiddenLayers = hp.Int(name="hiddenLayers", min_value=self.hiddenLayers["min"],
                                      max_value=self.hiddenLayers["max"], step=self.hiddenLayers["step"])
        return createNetwork(self.nCols, self.mlpName, self.hp_hiddenLayers, self.dropoutRate, self.seqLength, self.eta)
def createNetwork(nCols, mlpName, hiddenLayers=128, dropoutRate=1/2**2, seqLength=5, eta=2e-3):
    # nCols = 10
    # hiddenLayers = 128
    # dropoutRate = 0.2
    # seqLength = 5
    # eta = 1e-3

    if mlpName == "MLP_Desc_V1":
        # === input layers ===
        B0_input = layers.Input(shape=nCols, dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_dense = layers.Dense(units=hiddenLayers, name="B1_dense")(B0_embedding)
        B1_activation = layers.PReLU(name="B1_activation")(B1_dense)
        B1_dropout = layers.Dropout(rate=dropoutRate, name="B1_dropout")(B1_activation)

        B2_dense = layers.Dense(units=hiddenLayers, name="B2_dense")(B1_dropout)
        B2_activation = layers.PReLU(name="B2_activation")(B2_dense)
        B2_dropout = layers.Dropout(rate=dropoutRate, name="B2_dropout")(B2_activation)

        B3_dense = layers.Dense(units=hiddenLayers / 2, name="B3_dense")(B2_dropout)
        B3_activation = layers.PReLU(name="B3_activation")(B3_dense)
        B3_dropout = layers.Dropout(rate=dropoutRate, name="B3_dropout")(B3_activation)

        B4_dense = layers.Dense(units=hiddenLayers / 2, name="B4_dense")(B3_dropout)
        B4_activation = layers.PReLU(name="B4_activation")(B4_dense)
        B4_dropout = layers.Dropout(rate=dropoutRate, name="B4_dropout")(B4_activation)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 4), activation="relu", name="layer_final")(B4_dropout)
    elif mlpName == "MLP_ResNet_V1":
        # === input layers ===
        B0_input = layers.Input(shape=nCols, dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_dense = layers.Dense(units=hiddenLayers, name="B1_dense")(B0_embedding)
        B1_activation = layers.PReLU(name="B1_activation")(B1_dense)

        B2_dense = layers.Dense(units=hiddenLayers, name="B2_dense")(B1_activation)
        B2_activation = layers.PReLU(name="B2_activation")(B2_dense)

        B3_dense = layers.Dense(units=hiddenLayers, name="B3_dense")(B2_activation)
        B1_B3_SkipConnection = layers.Add(name="B1_B3_SkipConnection")([B1_activation, B3_dense])
        B3_activation = layers.PReLU(name="B3_activation")(B1_B3_SkipConnection)
        B3_dropout = layers.Dropout(rate=dropoutRate, name="B3_dropout")(B3_activation)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B3_dropout)
    elif mlpName == "MLP_DenseNet_V1":
        # === input layers ===
        B0_input = layers.Input(shape=nCols, dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_dense = layers.Dense(units=hiddenLayers, name="B1_dense")(B0_embedding)
        B1_activation = layers.PReLU(name="B1_activation")(B1_dense)
        B1_concat = layers.Concatenate(name="B1_concat")([B0_embedding, B1_activation])

        B2_dense = layers.Dense(units=hiddenLayers, name="B2_dense")(B1_concat)
        B2_activation = layers.PReLU(name="B2_activation")(B2_dense)
        B2_concat = layers.Concatenate(name="B2_concat")([B0_embedding, B1_activation, B2_activation])

        B3_dense = layers.Dense(units=hiddenLayers, name="B3_dense")(B2_concat)
        B3_activation = layers.PReLU(name="B3_activation")(B3_dense)
        B3_dropout = layers.Dropout(rate=dropoutRate, name="B3_dropout")(B3_activation)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B3_dropout)
    elif mlpName == "MLP_LP_V1":
        # Source : Kaggle - Laurent Pourchot
        # URL : https://www.kaggle.com/pourchot/simple-neural-network

        # === input layers ===
        B0_input = layers.Input(shape=nCols, dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_dense = tfa.layers.WeightNormalization(
            layers.Dense(
                units=hiddenLayers, activation="selu"), name="B1_dense"
        )(B0_embedding)
        B1_dropout = layers.Dropout(rate=dropoutRate, name="B1_dropout")(B1_dense)
        B1_concat = layers.Concatenate(name="B1_concat")([B0_embedding, B1_dropout])

        B2_dense = tfa.layers.WeightNormalization(
            layers.Dense(
                units=hiddenLayers, activation="relu"), name="B2_dense"
        )(B1_concat)
        B2_dropout = layers.Dropout(rate=dropoutRate, name="B2_dropout")(B2_dense)
        B2_concat = layers.Concatenate(name="B2_concat")([B0_embedding, B1_dropout, B2_dropout])

        B3_dense = tfa.layers.WeightNormalization(
            layers.Dense(
                units=hiddenLayers, activation="elu"), name="B3_dense"
        )(B2_concat)
        B3_dropout = layers.Dropout(rate=dropoutRate, name="B3_dropout")(B3_dense)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B3_dropout)
    elif mlpName == "MLP_MultiActs_V1":
        # === input layers ===
        B0_input = layers.Input(shape=nCols, dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_dense1 = layers.Dense(units=int(hiddenLayers / 4), name="B1_dense1")(B0_embedding)
        B1_activation1 = activations.relu(B1_dense1)
        B1_dense2 = layers.Dense(units=int(hiddenLayers / 4), name="B1_dense2")(B0_embedding)
        B1_activation2 = activations.selu(B1_dense2)
        B1_dense3 = layers.Dense(units=int(hiddenLayers / 4), name="B1_dense3")(B0_embedding)
        B1_activation3 = activations.softplus(B1_dense3)
        B1_dense4 = layers.Dense(units=int(hiddenLayers / 4), name="B1_dense4")(B0_embedding)
        B1_activation4 = activations.gelu(B1_dense4)
        B1_concat1 = layers.concatenate([B1_activation1, B1_activation2, B1_activation3, B1_activation4],
                                        name="B1_concat")
        B1_dropout = layers.Dropout(rate=dropoutRate, name="B1_dropout")(B1_concat1)
        B1_concat2 = layers.concatenate([B0_embedding, B1_dropout], name="B1_concat2")

        B2_dense1 = layers.Dense(units=int(hiddenLayers / 4), name="B2_dense1")(B1_concat2)
        B2_activation1 = activations.relu(B2_dense1)
        B2_dense2 = layers.Dense(units=int(hiddenLayers / 4), name="B2_dense2")(B1_concat2)
        B2_activation2 = activations.selu(B2_dense2)
        B2_dense3 = layers.Dense(units=int(hiddenLayers / 4), name="B2_dense3")(B1_concat2)
        B2_activation3 = activations.softplus(B2_dense3)
        B2_dense4 = layers.Dense(units=int(hiddenLayers / 4), name="B2_dense4")(B1_concat2)
        B2_activation4 = activations.gelu(B2_dense4)
        B2_concat1 = layers.concatenate([B2_activation1, B2_activation2, B2_activation3, B2_activation4],
                                        name="B2_concat")
        B2_dropout = layers.Dropout(rate=dropoutRate, name="B2_dropout")(B2_concat1)
        B2_concat2 = layers.concatenate([B0_embedding, B1_dropout, B2_dropout], name="B2_concat2")

        B3_dense1 = layers.Dense(units=int(hiddenLayers / 4), name="B3_dense1")(B2_concat2)
        B3_activation1 = activations.relu(B3_dense1)
        B3_dense2 = layers.Dense(units=int(hiddenLayers / 4), name="B3_dense2")(B2_concat2)
        B3_activation2 = activations.selu(B3_dense2)
        B3_dense3 = layers.Dense(units=int(hiddenLayers / 4), name="B3_dense3")(B2_concat2)
        B3_activation3 = activations.softplus(B3_dense3)
        B3_dense4 = layers.Dense(units=int(hiddenLayers / 4), name="B3_dense4")(B2_concat2)
        B3_activation4 = activations.gelu(B3_dense4)
        B3_concat1 = layers.concatenate([B3_activation1, B3_activation2, B3_activation3, B3_activation4],
                                        name="B3_concat")
        B3_dropout = layers.Dropout(rate=dropoutRate, name="B3_dropout")(B3_concat1)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B3_dropout)
    elif mlpName == "MLP_LSTM_V1":
        # === input layers ===
        B0_input = layers.Input(shape=(seqLength, nCols), dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate, return_sequences=True)(B0_embedding)
        B2_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate)(B1_lstm)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 2), activation="relu", name="layer_final")(B2_lstm)
    elif mlpName == "MLP_LSTM_V2":
        # === input layers ===
        B0_input = layers.Input(shape=(seqLength, nCols), dtype="float32", name="B0_input")
        B0_embedding = layers.Dense(units=hiddenLayers * 2, activation="relu",
                                    kernel_regularizer="l2", name="B0_embedding")(B0_input)

        # === learning layers ===
        B1_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate, return_sequences=True)(B0_embedding)
        B2_lstm = layers.LSTM(units=hiddenLayers, dropout=dropoutRate, return_sequences=True)(B1_lstm)
        B3_lstm = layers.LSTM(units=int(hiddenLayers / 2), dropout=dropoutRate, return_sequences=True)(B2_lstm)
        B4_lstm = layers.LSTM(units=int(hiddenLayers / 2), dropout=dropoutRate)(B3_lstm)

        # === top layers ===
        layer_final = layers.Dense(units=int(hiddenLayers / 4), activation="relu", name="layer_final")(B4_lstm)
    elif mlpName == "MLP_CatEmbedding":
        # ----------- Embedding layers ----------------------
        B0_input = layers.Input(shape=(nCols), name="B0_input")
        B0_embedding = layers.Embedding(input_dim=512,
                                        output_dim=16,
                                        name="B0_embedding")(B0_input)
        # ----------- Convolution layers ----------------------
        B1_conv1d = layers.Conv1D(4, 1, activation='relu', name="B1_conv1d")(B0_embedding)
        B1_flatten = layers.Flatten(name="extract")(B1_conv1d)
        layer_final = layers.Dropout(0.25, name="layer_final")(B1_flatten)
    else:
        print("Available Model list --->", ["MLP_Desc_V1", "MLP_ResNet_V1", "MLP_DenseNet_V1", "MLP_LP_V1"])
        return None

    if targetType == "numeric":
        layer_regressor = layers.Dense(units=1, name="Regressor")(layer_final)
        model_mlp = Model(B0_input, layer_regressor)
        model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="mean_squared_error",
                          metrics=tf_metrics.RootMeanSquaredError(name="rmse"))
    else:
        if targetTask == "binary":
            layer_classifier = layers.Dense(units=1, activation="sigmoid", name="Classifier")(layer_final)
            model_mlp = Model(B0_input, layer_classifier)
            model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="binary_crossentropy",
                              metrics=tf_metrics.AUC(name="roc_auc"))
        else:
            layer_classifier = layers.Dense(units=len(class_levels), activation="softmax", name="Classifier")(
                layer_final)
            model_mlp = Model(B0_input, layer_classifier)
            model_mlp.compile(optimizer=optimizers.Adam(learning_rate=eta), loss="categorical_crossentropy",
                              metrics=tf_metrics.CategoricalAccuracy(name="accuracy"))

    return model_mlp
# # "Available Model list --->", ["MLP_Desc_V1", "MLP_ResNet_V1", "MLP_DenseNet_V1", "MLP_LP_V1", "MLP_MultiActs_V1"]
# targetType = "numeric"
# createNetwork(nCols=10, mlpName="MLP_MultiActs_V2").summary()
def doMLP(train_x, train_y, test_x, test_y, mlpName="MLP_Desc_V1", seqLength=None,
          hiddenLayers={"min": 32, "max": 128, "step": 32}, dropoutRate=1/2**2,
          epochs=10, batch_size=32, model_export=False, preTrained=None, seed=515):
    result_dic = {}
    scaler_minmax = prep.MinMaxScaler()
    train_x = scaler_minmax.fit_transform(train_x)
    test_x = scaler_minmax.transform(test_x)
    patientRate = 0.2
    tf.random.set_seed(seed)
    cb_earlystopping = tf_callbacks.EarlyStopping(patience=int(epochs * patientRate),
                                                  restore_best_weights=True)
    cb_reduceLR = tf_callbacks.ReduceLROnPlateau(patience=int((epochs * patientRate)/10), factor=0.8, min_lr=1e-4)
    cb_lists = [cb_earlystopping, cb_reduceLR, TqdmCallback(verbose=0)]

    runStart = time()
    if targetType == "numeric":
        # LSTM model
        if seqLength is not None:
            train_ts_dataset = make_ts_tensor(train_x, train_y, sequence_length=seqLength, batch_size=batch_size)
            test_ts_dataset = make_ts_tensor(test_x, test_y, sequence_length=seqLength, batch_size=batch_size)
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if type(hiddenLayers) is dict:
                    model_tuner = kt.BayesianOptimization(
                        MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4, seqLength=seqLength,
                                     hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                        objective="val_loss",
                        max_trials=int(epochs * 0.2),
                        seed=seed + 1,
                        overwrite=True)
                    model_tuner.search(train_ts_dataset, validation_data=test_ts_dataset, shuffle=False)

                    result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                    print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])
                else:
                    result_dic["best_params"] = {"hiddenLayers": hiddenLayers}

                result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                    hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                    dropoutRate=dropoutRate)

                result_dic["best_params"]["epochs"] = result_dic['model'].fit(train_ts_dataset,
                                                                              epochs=epochs,
                                                                              validation_data=test_ts_dataset,
                                                                              verbose=0, shuffle=False,
                                                                              callbacks=cb_lists)
                result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

            if test_x is not None:
                result_dic["pred"] = result_dic["model"].predict(test_ts_dataset)
                if test_y is not None:
                    mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                    rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                    result_dic["performance"] = {"MAE": mae,
                                                 "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                                 "NMAE": mae / test_y.abs().mean(),
                                                 "RMSE": rmse,
                                                 "NRMSE": rmse / test_y.abs().mean(),
                                                 "R2": metrics.r2_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None
        # not LSTM
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                if type(hiddenLayers) is dict:
                    model_tuner = kt.BayesianOptimization(
                        MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4,
                                     hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                        objective="val_loss",
                        max_trials=int(epochs * 0.2),
                        seed=seed + 1,
                        overwrite=True)

                    model_tuner.search(train_x, train_y, batch_size=batch_size,
                                       validation_data=(test_x, test_y),
                                       shuffle=False)

                    result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                    print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])
                else:
                    result_dic["best_params"] = {"hiddenLayers": hiddenLayers}

                result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                    hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                    dropoutRate=dropoutRate)

                result_dic["best_params"]["epochs"] = result_dic['model'].fit(x=train_x, y=train_y,
                                                                              epochs=epochs, batch_size=batch_size,
                                                                              validation_data=(test_x, test_y),
                                                                              verbose=0, shuffle=False,
                                                                              callbacks=cb_lists)
                result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

            if test_x is not None:
                result_dic["pred"] = result_dic["model"].predict(test_x, batch_size=batch_size)
                if test_y is not None:
                    mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                    rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                    result_dic["performance"] = {"MAE": mae,
                                                 "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                                 "NMAE": mae / test_y.abs().mean(),
                                                 "RMSE": rmse,
                                                 "NRMSE": rmse / test_y.abs().mean(),
                                                 "R2": metrics.r2_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if seqLength is not None:
                train_ts_dataset = make_ts_tensor(train_x, train_y, sequence_length=seqLength, batch_size=batch_size)
                test_ts_dataset = make_ts_tensor(test_x, test_y, sequence_length=seqLength, batch_size=batch_size)

                if preTrained is not None:
                    result_dic["model"] = preTrained
                else:
                    if type(hiddenLayers) is dict:
                        model_tuner = kt.BayesianOptimization(
                            MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4, seqLength=seqLength,
                                         hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                            objective="val_loss",
                            max_trials=int(epochs * 0.2),
                            seed=seed + 1,
                            overwrite=True)
                        model_tuner.search(train_ts_dataset, validation_data=test_ts_dataset, shuffle=False)

                        result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                        print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])
                    else:
                        result_dic["best_params"] = {"hiddenLayers": hiddenLayers}

                    result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                        hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                        dropoutRate=dropoutRate)

                    result_dic["best_params"]["epochs"] = result_dic['model'].fit(train_ts_dataset,
                                                                                  epochs=epochs,
                                                                                  validation_data=test_ts_dataset,
                                                                                  verbose=0, shuffle=False,
                                                                                  callbacks=cb_lists)
                    result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

                if test_x is not None:
                    result_dic["pred"] = result_dic["model"].predict(test_ts_dataset)
                    if test_y is not None:
                        mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                        rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                        result_dic["performance"] = {"MAE": mae,
                                                     "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                                     "NMAE": mae / test_y.abs().mean(),
                                                     "RMSE": rmse,
                                                     "NRMSE": rmse / test_y.abs().mean(),
                                                     "R2": metrics.r2_score(test_y, result_dic["pred"])}
                    else:
                        result_dic["performance"] = None
            else:
                if preTrained is not None:
                    result_dic["model"] = preTrained
                else:
                    model_tuner = kt.BayesianOptimization(
                        MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4,
                                     hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                        objective="val_loss",
                        max_trials=int(epochs * 0.2),
                        seed=seed + 1,
                        overwrite=True,
                    )

                    model_tuner.search(train_x, train_y, batch_size=batch_size,
                                       validation_data=(test_x, test_y),
                                       shuffle=False)

                    result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                    print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])

                    result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                        hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                        dropoutRate=dropoutRate)

                    result_dic["best_params"]["epochs"] = result_dic['model'].fit(x=train_x, y=train_y,
                                                                                  epochs=epochs, batch_size=batch_size,
                                                                                  validation_data=(test_x, test_y),
                                                                                  verbose=0, shuffle=False,
                                                                                  callbacks=cb_lists)
                    result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

                if test_x is not None:
                    result_dic["prob"] = result_dic["model"].predict(test_x, batch_size=batch_size)
                    result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                    if test_y is not None:
                        result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                     "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                     "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                     "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                    else:
                        result_dic["performance"] = None
        else: # multiclass
            encoder_onehot = OneHotEncoder(sparse=False)
            train_y_sparse = encoder_onehot.fit_transform(train_y[..., np.newaxis])
            test_y_sparse = encoder_onehot.transform(test_y[..., np.newaxis])
            if seqLength is not None:
                train_ts_dataset = make_ts_tensor(train_x, train_y_sparse, sequence_length=seqLength, batch_size=batch_size)
                test_ts_dataset = make_ts_tensor(test_x, test_y_sparse, sequence_length=seqLength, batch_size=batch_size)

                if preTrained is not None:
                    result_dic["model"] = preTrained
                else:
                    if type(hiddenLayers) is dict:
                        model_tuner = kt.BayesianOptimization(
                            MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4, seqLength=seqLength,
                                         hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                            objective="val_loss",
                            max_trials=int(epochs * 0.2),
                            seed=seed + 1,
                            overwrite=True)
                        model_tuner.search(train_ts_dataset, validation_data=test_ts_dataset, shuffle=False)

                        result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                        print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])
                    else:
                        result_dic["best_params"] = {"hiddenLayers": hiddenLayers}

                    result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                        hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                        dropoutRate=dropoutRate)

                    result_dic["best_params"]["epochs"] = result_dic['model'].fit(train_ts_dataset,
                                                                                  epochs=epochs,
                                                                                  validation_data=test_ts_dataset,
                                                                                  verbose=0, shuffle=False,
                                                                                  callbacks=cb_lists)
                    result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

                if test_x is not None:
                    result_dic["prob"] = result_dic["model"].predict(test_x, batch_size=batch_size)
                    # axis=0 ---> column, axis=1 ---> row
                    result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                    if test_y is not None:
                        result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                     "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                    else:
                        result_dic["performance"] = None
            else:
                if preTrained is not None:
                    result_dic["model"] = preTrained
                else:
                    model_tuner = kt.BayesianOptimization(
                        MyHyperModel(train_x.shape[1], mlpName=mlpName, eta=1e-4,
                                     hiddenLayers=hiddenLayers, dropoutRate=dropoutRate),
                        objective="val_loss",
                        max_trials=int(epochs * 0.2),
                        seed=seed + 1,
                        overwrite=True,
                    )

                    model_tuner.search(train_x, train_y_sparse, batch_size=batch_size,
                                       validation_data=(test_x, test_y_sparse),
                                       shuffle=False)

                    result_dic["best_params"] = {"hiddenLayers": model_tuner.get_best_hyperparameters()[0].get("hiddenLayers")}
                    print("\nTuning Result ---> Hidden layers :", result_dic["best_params"]["hiddenLayers"])

                    result_dic['model'] = createNetwork(nCols=train_x.shape[1], mlpName=mlpName,
                                                        hiddenLayers=result_dic["best_params"]["hiddenLayers"],
                                                        dropoutRate=dropoutRate)

                    result_dic["best_params"]["epochs"] = result_dic['model'].fit(x=train_x, y=train_y_sparse,
                                                                                  epochs=epochs, batch_size=batch_size,
                                                                                  validation_data=(test_x, test_y_sparse),
                                                                                  verbose=0, shuffle=False,
                                                                                  callbacks=cb_lists)
                    result_dic["best_params"]["epochs"] = np.argmin(result_dic["best_params"]["epochs"].history["val_loss"])

                if test_x is not None:
                    result_dic["prob"] = result_dic["model"].predict(test_x, batch_size=batch_size)
                    # axis=0 ---> column, axis=1 ---> row
                    result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                    if test_y is not None:
                        result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                     "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                    else:
                        result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic

# input_epochs = 200
# input_batch_size = 4
#
# input_hiddenLayers = {"min": 64, "max": 512, "step": 64}
#
# # === MLP_Desc_V1 ===
# result_val["MLP_Desc_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
#                                   ds["val_x_oh"], ds["val_y"],
#                                   mlpName="MLP_Desc_V1", epochs=input_epochs, batch_size=input_batch_size,
#                                   hiddenLayers=input_hiddenLayers, model_export=True)
# print(result_val["MLP_Desc_V1"]["model"].summary())
# print(result_val["MLP_Desc_V1"]["best_params"])
# print(result_val["MLP_Desc_V1"]["performance"])
# print(result_val["MLP_Desc_V1"]["running_time"])
#
# result_test["MLP_Desc_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
#                                    ds["test_x_oh"], None,
#                                    preTrained=result_val["MLP_Desc_V1"]["model"])
# print(result_val["MLP_Desc_V1"]["pred"][:10])
#
# # === MLP_ResNet_V1 ===
# result_val["MLP_ResNet_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
#                                     ds["val_x_oh"], ds["val_y"],
#                                     mlpName="MLP_ResNet_V1", epochs=input_epochs, batch_size=input_batch_size,
#                                     hiddenLayers=input_hiddenLayers, model_export=True)
# print(result_val["MLP_ResNet_V1"]["model"].summary())
# print(result_val["MLP_ResNet_V1"]["best_params"])
# print(result_val["MLP_ResNet_V1"]["performance"])
# print(result_val["MLP_ResNet_V1"]["running_time"])
#
# result_test["MLP_ResNet_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
#                                      ds["test_x_oh"], None,
#                                      preTrained=result_val["MLP_ResNet_V1"]["model"])
# print(result_val["MLP_ResNet_V1"]["pred"][:10])
#
# # === MLP_DenseNet_V1 ===
# result_val["MLP_DenseNet_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
#                                       ds["val_x_oh"], ds["val_y"],
#                                       mlpName="MLP_DenseNet_V1", epochs=input_epochs, batch_size=input_batch_size,
#                                       hiddenLayers=input_hiddenLayers, model_export=True)
# print(result_val["MLP_DenseNet_V1"]["model"].summary())
# print(result_val["MLP_DenseNet_V1"]["best_params"])
# print(result_val["MLP_DenseNet_V1"]["performance"])
# print(result_val["MLP_DenseNet_V1"]["running_time"])
#
# result_test["MLP_DenseNet_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
#                                        ds["test_x_oh"], None,
#                                        preTrained=result_val["MLP_DenseNet_V1"]["model"])
# print(result_val["MLP_DenseNet_V1"]["pred"][:10])
#
# # === MLP_LP_V1 ===
# result_val["MLP_LP_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
#                                 ds["val_x_oh"], ds["val_y"],
#                                 mlpName="MLP_LP_V1", epochs=input_epochs, batch_size=input_batch_size,
#                                 hiddenLayers=input_hiddenLayers, model_export=True)
# print(result_val["MLP_LP_V1"]["model"].summary())
# print(result_val["MLP_LP_V1"]["best_params"])
# print(result_val["MLP_LP_V1"]["performance"])
# print(result_val["MLP_LP_V1"]["running_time"])
#
# result_test["MLP_LP_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
#                                  ds["test_x_oh"], None,
#                                  preTrained=result_val["MLP_LP_V1"]["model"])
# print(result_val["MLP_LP_V1"]["pred"][:10])
#
# # === MLP_MultiActs_V1 ===
# result_val["MLP_MultiActs_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
#                                        ds["val_x_oh"], ds["val_y"],
#                                        mlpName="MLP_MultiActs_V1", epochs=input_epochs, batch_size=input_batch_size,
#                                        hiddenLayers=input_hiddenLayers, model_export=True)
# print(result_val["MLP_MultiActs_V1"]["model"].summary())
# print(result_val["MLP_MultiActs_V1"]["best_params"])
# print(result_val["MLP_MultiActs_V1"]["performance"])
# print(result_val["MLP_MultiActs_V1"]["running_time"])
#
# result_test["MLP_MultiActs_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
#                                         ds["test_x_oh"], None,
#                                         preTrained=result_val["MLP_MultiActs_V1"]["model"])
# print(result_val["MLP_MultiActs_V1"]["pred"][:10])



# # # save obejcts
# # easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# # easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")
#
# # display the performance
#
#
#
# # Stacking Ensemble
# seed_base_models = 933
#
# # if regression
# # result_val["MLP_Desc_V1"]["model"].trainable = False
# # result_val["MLP_ResNet_V1"]["model"].trainable = False
# # result_val["MLP_MultiActs_V1"]["model"].trainable = False
#
# # CAT_GBM   0.355392  0.837989  0.805369  0.920356
# # XGB_GBT   0.355467  0.854749  0.824324  0.917260
# # LGB_RF
# #
# # MLP_ResNet_V1        NaN  0.871508  0.836879  0.913570
# # MLP_LP_V1   0.368466  0.849162  0.805755  0.913307
# # MLP_MultiActs_V1   0.367828  0.871508  0.834532  0.912385
# # LGB_GOSS   0.377965  0.832402  0.794521  0.911331
#
# # result_val["MLP_ResNet_V1"]["model"].trainable = False
# # result_val["MLP_LP_V1"]["model"].trainable = False
# # result_val["MLP_MultiActs_V1"]["model"].trainable = False
# stacking_base_models = [
#     # ('ElasticNet', make_pipeline(StandardScaler(), lm.ElasticNet(alpha=result_val["ElasticNet"]["best_params"]["alpha"],
#     #                                                              l1_ratio=result_val["ElasticNet"]["best_params"]["l1_ratio"],
#     #                                                              normalize=False, random_state=seed_base_models+1))),
#     # ('SVM', make_pipeline(MinMaxScaler(), svm.SVR(kernel=result_val["SVM"]["best_params"]["kernel"],
#     #                                               C=result_val["SVM"]["best_params"]["C"],
#     #                                               gamma=result_val["SVM"]["best_params"]["gamma"]))),
#     ('XGB_GBT', xgb.XGBClassifier(booster="gbtree", objective="binary:logistic", learning_rate=5e-3,
#                                   n_estimators=result_val["XGB_GBT"]["model"].best_iteration,
#                                   max_depth=result_val["XGB_GBT"]["best_params"]["max_depth"],
#                                   subsample=result_val["XGB_GBT"]["best_params"]["subsample"],
#                                   reg_lambda=result_val["XGB_GBT"]["best_params"]["reg_lambda"],
#                                   min_child_weight=result_val["XGB_GBT"]["best_params"]["min_child_weight"],
#                                   gamma=result_val["XGB_GBT"]["best_params"]["gamma"],
#                                   colsample_bytree=0.8, verbosity=0, use_label_encoder=False,
#                                   n_jobs=None, random_state=seed_base_models+2)),
#     ('LGB_RF', lgb.LGBMClassifier(boosting_type="rf", objective="binary", learning_rate=5e-3,
#                                   n_estimators=result_val["LGB_RF"]["model"].best_iteration_,
#                                   num_leaves=result_val["LGB_RF"]["best_params"]["num_leaves"],
#                                   subsample=result_val["LGB_RF"]["best_params"]["subsample"],
#                                   reg_lambda=result_val["LGB_RF"]["best_params"]["reg_lambda"],
#                                   min_child_weight=result_val["LGB_RF"]["best_params"]["min_child_weight"],
#                                   min_child_samples=result_val["LGB_RF"]["best_params"]["min_child_samples"],
#                                   min_split_gain=result_val["LGB_RF"]["best_params"]["min_split_gain"],
#                                   subsample_freq=2, colsample_bytree=0.8, silent=True,
#                                   n_jobs=None, random_state=seed_base_models+3)),
#     # ('LGB_GOSS', lgb.LGBMClassifier(boosting_type="goss", objective="binary", learning_rate=5e-3,
#     #                                 n_estimators=result_val["LGB_GOSS"]["model"].best_iteration_,
#     #                                 num_leaves=result_val["LGB_GOSS"]["best_params"]["num_leaves"],
#     #                                 subsample=result_val["LGB_GOSS"]["best_params"]["subsample"],
#     #                                 reg_lambda=result_val["LGB_GOSS"]["best_params"]["reg_lambda"],
#     #                                 min_child_weight=result_val["LGB_GOSS"]["best_params"]["min_child_weight"],
#     #                                 min_child_samples=result_val["LGB_GOSS"]["best_params"]["min_child_samples"],
#     #                                 min_split_gain=result_val["LGB_GOSS"]["best_params"]["min_split_gain"],
#     #                                 colsample_bytree=0.8, silent=True,
#     #                                 n_jobs=None, random_state=seed_base_models+4)),
#     ('CAT_GBM', cat.CatBoostClassifier(boosting_type="Plain", loss_function="Logloss", learning_rate=5e-2,
#                                        n_estimators=result_val["CAT_GBM"]["model"].best_iteration_,
#                                        max_depth=result_val["CAT_GBM"]["best_params"]["max_depth"],
#                                        bagging_temperature=result_val["CAT_GBM"]["best_params"]["bagging_temperature"],
#                                        l2_leaf_reg=result_val["CAT_GBM"]["best_params"]["l2_leaf_reg"],
#                                        random_strength=result_val["CAT_GBM"]["best_params"]["random_strength"],
#                                        rsm=0.8, logging_level="Silent",
#                                        thread_count=None, random_state=seed_base_models+5)),
#     # ("MLP_ResNet_V1", KerasClassifier(model=result_val["MLP_ResNet_V1"]["model"], batch_size=4, shuffle=False,
#     #                                   verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50)),
#     # ("MLP_LP_V1", KerasClassifier(model=result_val["MLP_LP_V1"]["model"], batch_size=4, shuffle=False,
#     #                               verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50)),
#     # ("MLP_MultiActs_V1", KerasClassifier(model=result_val["MLP_MultiActs_V1"]["model"], batch_size=4, shuffle=False,
#     #                                      verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50))
# ]


# meta learner definition
# meta_learner_model = lm.LinearRegression()
# meta_learner_model = lm.ElasticNetCV(cv=10, alphas=np.linspace(1e-3, 1e+3, 100).tolist(), random_state=seed_base_models+100,
#                                      l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])
# meta_learner_model = lm.LogisticRegressionCV(penalty="elasticnet", solver="saga", multi_class="ovr", cv=10, max_iter=1000, random_state=4,
#                                              Cs=np.linspace(1e-3, 1e+3, 100).tolist(),
#                                              l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])




def doStackingEnsemble(train_x, train_y, test_x, test_y, stacking_base_models, meta_learner_model,
                       kfolds=KFold(10, shuffle=True, random_state=2323), model_export=False, preTrained=None, seed=7788):
    result_dic = {}
    rnd.seed(seed)

    runStart = time()
    if targetType == "numeric":
        if preTrained is not None:
            result_dic["model"] = preTrained
        else:
            result_dic["model"] = ensemble.StackingRegressor(estimators=stacking_base_models,
                                                             final_estimator=meta_learner_model,
                                                             cv=kfolds.split(train_x, train_y),
                                                             n_jobs=multiprocessing.cpu_count())
            result_dic["model"].fit(train_x, train_y)

        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = ensemble.StackingClassifier(estimators=stacking_base_models,
                                                                  final_estimator=meta_learner_model,
                                                                  cv=kfolds.split(train_x, train_y),
                                                                  stack_method="predict_proba",
                                                                  n_jobs=multiprocessing.cpu_count())
                result_dic["model"].fit(train_x, train_y)

            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                result_dic["prob"] = result_dic["prob"][:,1,np.newaxis]
                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:, 0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
                else:
                    result_dic["performance"] = None
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                result_dic["model"] = ensemble.StackingClassifier(estimators=stacking_base_models,
                                                                  cv=kfolds.split(train_x, train_y),
                                                                  stack_method="predict_proba",
                                                                  final_estimator=meta_learner_model,
                                                                  n_jobs=multiprocessing.cpu_count())
                result_dic["model"].fit(train_x, train_y)

            if test_x is not None:
                result_dic["prob"] = result_dic["model"].predict_proba(test_x)
                # axis=0 ---> column, axis=1 ---> row
                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}
                else:
                    result_dic["performance"] = None

    if (not model_export) or (preTrained is not None): result_dic["model"] = None
    result_dic["running_time"] = round(time() - runStart, 3)
    print(f"Running Time ---> {result_dic['running_time']} sec")
    return result_dic


# result_val["StackingEnsemble"] = doStackingEnsemble(ds["train_x_oh"], ds["train_y"],
#                                                     ds["val_x_oh"], ds["val_y"],
#                                                     stacking_base_models=stacking_base_models,
#                                                     meta_learner_model=meta_learner_model,
#                                                     kfolds=kfolds_spliter, model_export=True)
# print(result_val["StackingEnsemble"]["performance"])
#
#
# result_test["StackingEnsemble"] = doStackingEnsemble(ds["train_x_oh"], ds["train_y"],
#                                                      ds["test_x_oh"], None,
#                                                      stacking_base_models=None, meta_learner_model=None,
#                                                      preTrained=result_val["StackingEnsemble"]["model"])
# print(result_test["StackingEnsemble"]["pred"][:10])
#
# # # save obejcts
# # easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# # easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")
#
# # display the performance
# for k, v in result_val.items():
#     if v is not None:
#         if "performance" in v.keys():
#             print(k, "--->", v["performance"])


# K-folds MLP ensemble
class kfoldsMLP:
    def __init__(self, nCols, mlpName, hiddenLayers=64, dropoutRate=1/2**2, seqLength=5, boosters={}, meta_learner=None, ntrees=5000, earlyStopping=100):
        self.nCols = nCols
        self.scaler_minmax = MinMaxScaler()
        self.mlpName = mlpName
        self.hiddenLayers = hiddenLayers
        self.dropoutRate = dropoutRate
        self.seqLength = seqLength
        self.ensemble_pred = None
        self.pred = None
        self.predWB = None
        self.prob = None
        self.probWB = None
        self.mlp_finalscore = None
        self.performance = {}
        self.performanceWB = {}
        self.model_list = []
        self.boosters = boosters
        self.meta_learner = meta_learner
        self.ntrees = ntrees
        self.earlyStopping = earlyStopping
        self.running_time = 0
    def fit_predict(self, train_x, train_y, test_x, test_y, epochs=20, batch_size=32, kfolds=10, stratify=None,
                    model_export=False, withBoosting=False, seed=8765):
        train_x = self.scaler_minmax.fit_transform(train_x)
        test_x = self.scaler_minmax.transform(test_x)
        patientRate = 0.2
        tf.random.set_seed(seed)
        rnd.seed(seed+1)
        cb_earlystopping = tf_callbacks.EarlyStopping(patience=int(epochs * patientRate),
                                                      restore_best_weights=True)
        cb_reduceLR = tf_callbacks.ReduceLROnPlateau(patience=int((epochs * patientRate)/10),
                                                     factor=0.8, min_lr=1e-4)
        cb_lists = [cb_earlystopping, cb_reduceLR]

        runStart = time()
        if withBoosting:
            print("INFO : model_export is set to True (Since withBoosting is True)")
            model_export = True

        if targetType == "numeric":
            self.ensemble_pred = np.zeros((train_x.shape[0], 1))
            self.pred = np.zeros((test_x.shape[0], 1))
            if withBoosting:
                for i in self.boosters.keys():
                    self.boosters[i]["ensemble_pred"] = np.zeros((train_x.shape[0], 1))
                    self.boosters[i]["pred"] = np.zeros((test_x.shape[0], 1))
                    self.boosters[i]["model_list"] = []
        else:
            if targetTask == "binary":
                self.ensemble_prob = np.zeros((train_x.shape[0], 1))
                self.prob = np.zeros((test_x.shape[0], 1))
                if withBoosting:
                    for i in self.boosters.keys():
                        self.boosters[i]["ensemble_prob"] = np.zeros((train_x.shape[0], 1))
                        self.boosters[i]["prob"] = np.zeros((test_x.shape[0], 1))
                        self.boosters[i]["model_list"] = []
            else:
                self.ensemble_prob = np.zeros((train_x.shape[0], len(class_levels)))
                self.prob = np.zeros((test_x.shape[0], len(class_levels)))
                if withBoosting:
                    for i in self.boosters.keys():
                        self.boosters[i]["ensemble_prob"] = np.zeros((train_x.shape[0], len(class_levels)))
                        self.boosters[i]["prob"] = np.zeros((test_x.shape[0], len(class_levels)))
                        self.boosters[i]["model_list"] = []

        # ===== training kfolds model =====
        if stratify is None:
            skf = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
            stratify = train_y
        else:
            skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=seed)
        for fold, (nonkIdx, kIdx) in enumerate(skf.split(train_x, stratify)):
            print(f"\n ====== TRAINING FOLD {fold} =======")

            train_nonk_x = train_x[nonkIdx, ...]
            train_nonk_y = train_y.iloc[nonkIdx]
            train_k_x = train_x[kIdx, ...]
            train_k_y = train_y.iloc[kIdx]

            if targetType == "numeric":
                model_kfolds = createNetwork(self.nCols, self.mlpName, self.hiddenLayers, self.dropoutRate,
                                             self.seqLength)
                model_kfolds.fit(train_nonk_x, train_nonk_y,
                                 epochs=epochs, batch_size=batch_size,
                                 validation_data=(train_k_x, train_k_y),
                                 callbacks=cb_lists, shuffle=False,
                                 verbose=0)
                # Kth fold prediction
                pred_a = model_kfolds.predict(train_k_x, batch_size=batch_size)
                self.ensemble_pred[kIdx] += pred_a
                score_NN_a = np.sqrt(metrics.mean_squared_error(train_k_y, pred_a))
                print(f"\nMLP FOLD {fold} Score : {score_NN_a}\n")
                # test prediction
                self.pred += model_kfolds.predict(test_x, batch_size=batch_size) / kfolds
            else:
                if targetTask == "binary":
                    model_kfolds = createNetwork(self.nCols, self.mlpName, self.hiddenLayers, self.dropoutRate,
                                                 self.seqLength)
                    model_kfolds.fit(train_nonk_x, train_nonk_y,
                                     epochs=epochs, batch_size=batch_size,
                                     validation_data=(train_k_x, train_k_y),
                                     callbacks=cb_lists, shuffle=False,
                                     verbose=0)
                    # Kth fold prediction
                    pred_a = model_kfolds.predict(train_k_x, batch_size=batch_size)
                    self.ensemble_prob[kIdx] += pred_a
                    score_NN_a = metrics.log_loss(train_k_y, pred_a)
                    print(f"\nMLP FOLD {fold} Score: {score_NN_a}\n")
                    # test prediction
                    self.prob += model_kfolds.predict(test_x, batch_size=batch_size) / kfolds
                else:
                    encoder_onehot = OneHotEncoder(sparse=False)
                    train_nonk_y_sparse = encoder_onehot.fit_transform(train_nonk_y[..., np.newaxis])
                    train_k_y_sparse = encoder_onehot.transform(train_k_y[..., np.newaxis])

                    model_kfolds = createNetwork(self.nCols, self.mlpName, self.hiddenLayers, self.dropoutRate,
                                                 self.seqLength)
                    model_kfolds.fit(train_nonk_x, train_nonk_y_sparse,
                                     epochs=epochs, batch_size=batch_size,
                                     validation_data=(train_k_x, train_k_y_sparse),
                                     callbacks=cb_lists, shuffle=False,
                                     verbose=0)
                    # Kth fold prediction and performance
                    pred_a = model_kfolds.predict(train_k_x, batch_size=batch_size)
                    self.ensemble_prob[kIdx] += pred_a
                    score_NN_a = metrics.log_loss(train_k_y, pred_a)
                    print(f"\nMLP FOLD {fold} Score: {score_NN_a}\n")
                    # test prediction
                    self.prob += model_kfolds.predict(test_x, batch_size=batch_size) / kfolds

            # Boosting prediction
            if withBoosting:
                if len(self.boosters.keys()) == 0:
                    pass
                else:
                    mlp_nonHeader = Model(model_kfolds.get_layer("B0_input").output,
                                          model_kfolds.get_layer("layer_final").output)
                    self.mlpBoosting(dataframe(mlp_nonHeader.predict(train_nonk_x, batch_size=batch_size)), train_nonk_y,
                                     dataframe(mlp_nonHeader.predict(train_k_x, batch_size=batch_size)), train_k_y,
                                     dataframe(mlp_nonHeader.predict(test_x, batch_size=batch_size)), kIdx, kfolds)

            if model_export: self.model_list.append(model_kfolds)

        # ===== output final score and predict on test set =====
        # self.performanceWB : save order -> MLP, booster_1, booster_2 ... booster_N
        if withBoosting:
            # 1. save mlp and boosters performance
            # 2. apply meta learner to prediction
            if targetType == "numeric":
                self.performanceWB["MLP"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, self.pred)),
                                             "R2": metrics.r2_score(test_y, self.pred)}
                for k, v in self.boosters.items():
                    self.performanceWB[k] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, v["pred"])),
                                             "R2": metrics.r2_score(test_y, v["pred"])}
                print("===== MLP with Boosting Score By Models =====")
                print(self.performanceWB)
                # meta learning
                kfolds_train_pred = [self.ensemble_pred]
                test_pred = [self.pred]
                for booster_obj in self.boosters.values():
                    kfolds_train_pred.append(booster_obj["ensemble_pred"])
                    test_pred.append(booster_obj["pred"])

                kfolds_train_pred = np.concatenate(kfolds_train_pred, axis=1)
                test_pred = np.concatenate(test_pred, axis=1)
                self.meta_learner.fit(kfolds_train_pred, train_y)
                self.predWB = self.meta_learner.predict(test_pred)
                if test_y is not None:
                    mae = metrics.mean_absolute_error(test_y, self.predWB)
                    rmse = metrics.mean_squared_error(test_y, self.predWB, squared=False)
                    self.performanceWB["Total"] = {"MAE": mae,
                                                   "MAPE": metrics.mean_absolute_percentage_error(test_y, self.predWB),
                                                   "NMAE": mae / test_y.abs().mean(),
                                                   "RMSE": rmse,
                                                   "NRMSE": rmse / test_y.abs().mean(),
                                                   "R2": metrics.r2_score(test_y, self.predWB)}

                    print("===== MLP with Boosting Final Score =====")
                    print(self.performanceWB["Total"])
            else:
                if targetTask == "binary":
                    tmp = [self.prob]
                    tmp_pred = [1 if i >= cut_off else 0 for i in tmp[-1][:, 0]]
                    self.performanceWB["MLP"] = {"Logloss": metrics.log_loss(test_y, tmp[-1]),
                                                 "Accuracy": metrics.accuracy_score(test_y, tmp_pred),
                                                 "F1": metrics.f1_score(test_y, tmp_pred),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, tmp[-1])}
                    for k, v in self.boosters.items():
                        tmp.append(v["prob"])
                        tmp_pred = [1 if i >= cut_off else 0 for i in tmp[-1][:, 0]]
                        self.performanceWB[k] = {"Logloss": metrics.log_loss(test_y, tmp[-1]),
                                                 "Accuracy": metrics.accuracy_score(test_y, tmp_pred),
                                                 "F1": metrics.f1_score(test_y, tmp_pred),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, tmp[-1])}
                    print("===== MLP with Boosting Score By Models =====")
                    print(self.performanceWB)
                    # meta learning
                    kfolds_train_prob = [self.ensemble_prob]
                    test_prob = [self.prob]
                    for booster_obj in self.boosters.values():
                        kfolds_train_prob.append(booster_obj["ensemble_prob"])
                        test_prob.append(booster_obj["prob"])

                    kfolds_train_prob = np.concatenate(kfolds_train_prob, axis=1)
                    test_prob = np.concatenate(test_prob, axis=1)

                    self.meta_learner.fit(kfolds_train_prob, train_y)
                    self.probWB = self.meta_learner.predict_proba(test_prob)[..., 1, np.newaxis]
                    self.predWB = [1 if i >= cut_off else 0 for i in self.probWB[:, 0]]
                    if test_y is not None:
                        self.performanceWB["Total"] = {"Logloss": metrics.log_loss(test_y, self.probWB),
                                                       "Accuracy": metrics.accuracy_score(test_y, self.predWB),
                                                       "F1": metrics.f1_score(test_y, self.predWB),
                                                       "ROC_AUC": metrics.roc_auc_score(test_y, self.probWB)}
                        print("===== MLP with Boosting Final Score =====")
                        print(self.performanceWB["Total"])
                else:
                    tmp = [self.prob[..., np.newaxis]]
                    tmp_pred = tmp[-1].argmax(axis=1)
                    self.performanceWB["MLP"] = {"Logloss": metrics.log_loss(test_y, tmp[-1]),
                                                 "Accuracy": metrics.accuracy_score(test_y, tmp_pred)}
                    for k, v in self.boosters.items():
                        tmp.append(i["prob"])
                        tmp_pred = tmp[-1].argmax(axis=1)
                        self.performanceWB[k] = {"Logloss": metrics.log_loss(test_y, tmp[-1]),
                                                 "Accuracy": metrics.accuracy_score(test_y, tmp_pred)}

                    print("===== MLP with Boosting Score By Models =====")
                    print(self.performanceWB)

                    kfolds_train_prob = [self.ensemble_pred[..., np.newaxis]]
                    test_prob = [self.pred]
                    for booster_obj in self.boosters.values():
                        kfolds_train_prob.append(booster_obj["ensemble_prob"][..., np.newaxis])
                        test_prob.append(booster_obj["prob"][..., np.newaxis])

                    kfolds_train_prob = np.concatenate(kfolds_train_prob, axis=1).mean(axis=1)[..., np.newaxis]
                    test_prob = np.concatenate(test_prob, axis=1).mean(axis=1)[..., np.newaxis]

                    self.meta_learner.fit(kfolds_train_prob, train_y)
                    self.probWB = self.meta_learner.predict_proba(test_prob)
                    self.predWB = self.probWB.argmax(axis=1)
                    if test_y is not None:
                        self.performanceWB["Total"] = {"Logloss": metrics.log_loss(test_y, self.probWB),
                                                       "Accuracy": metrics.accuracy_score(test_y, self.predWB)}
                        print("===== MLP with Boosting Final Score =====")
                        print(self.performanceWB["Total"])
        else:
            if targetType == "numeric":
                score_a = np.sqrt(metrics.mean_squared_error(train_y, self.ensemble_pred))
                print(f"\n=== FINAL SCORE (training) : {score_a}===\n")
                self.mlp_finalscore = score_a
                if test_y is not None:
                    mae = metrics.mean_absolute_error(test_y, self.pred)
                    rmse = metrics.mean_squared_error(test_y, self.pred, squared=False)
                    self.performance = {"MAE": mae,
                                        "MAPE": metrics.mean_absolute_percentage_error(test_y, self.pred),
                                        "NMAE": mae / test_y.abs().mean(),
                                        "RMSE": rmse,
                                        "NRMSE": rmse / test_y.abs().mean(),
                                        "R2": metrics.r2_score(test_y, self.pred)}
            else:
                if targetTask == "binary":
                    score_a = metrics.log_loss(train_y, self.ensemble_prob)
                    print(f"\n=== FINAL SCORE (training) : {score_a}===\n")
                    self.mlp_finalscore = score_a
                    self.pred = [1 if i >= cut_off else 0 for i in self.prob[:, 0]]
                    if test_y is not None:
                        self.performance = {"Logloss": metrics.log_loss(test_y, self.prob),
                                            "Accuracy": metrics.accuracy_score(test_y, self.pred),
                                            "F1": metrics.f1_score(test_y, self.pred),
                                            "ROC_AUC": metrics.roc_auc_score(test_y, self.prob)}
                else:
                    score_a = metrics.log_loss(train_y, self.ensemble_prob)
                    print(f"\n=== FINAL SCORE (training) : {score_a}===\n")
                    self.mlp_finalscore = score_a
                    self.pred = self.prob.argmax(axis=1)
                    if test_y is not None:
                        self.performance = {"Logloss": metrics.log_loss(test_y, self.prob),
                                            "Accuracy": metrics.accuracy_score(test_y, self.pred)}

        self.running_time = round(time() - runStart, 3)
        print(f"Running Time ---> {self.running_time} sec")
    def predict(self, test_x, test_y, batch_size=32, withBoosting=False):
        if self.model_list is None:
            print("ERROR : first fitting the model with 'model_export=True'")
            return None

        result_dic = {}
        test_x = self.scaler_minmax.transform(test_x)

        if targetType == "numeric":
            result_dic["pred"] = np.zeros((test_x.shape[0], 1))
            for model_mlp in self.model_list:
                result_dic["pred"] += model_mlp.predict(test_x, batch_size=batch_size) / len(self.model_list)

            if withBoosting:
                # iteration on all boosters
                pred_concat = [result_dic["pred"][...,np.newaxis]]
                for booster in self.boosters.values():
                    # iteration on each models (best iteration parameter is not set)
                    booster_pred = np.zeros((test_x.shape[0], 1))
                    for i in booster["model_list"]:
                        booster_pred += i.predict(test_x) / len(booster["model_list"])
                    pred_concat.append(booster_pred[...,np.newaxis])
                pred_concat = np.concatenate(pred_concat, axis=1)
                result_dic["pred"] = self.meta_learner.predict(pred_concat)[...,np.newaxis]

            result_dic["pred"] = result_dic["pred"].flatten()
            if test_y is not None:
                mae = metrics.mean_absolute_error(test_y, result_dic["pred"])
                rmse = metrics.mean_squared_error(test_y, result_dic["pred"], squared=False)
                result_dic["performance"] = {"MAE": mae,
                                             "MAPE": metrics.mean_absolute_percentage_error(test_y, result_dic["pred"]),
                                             "NMAE": mae / test_y.abs().mean(),
                                             "RMSE": rmse,
                                             "NRMSE": rmse / test_y.abs().mean(),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
        else:
            if targetTask == "binary":
                result_dic["prob"] = np.zeros((test_x.shape[0], 1))
                #
                for model_mlp in self.model_list:
                    result_dic["prob"] += model_mlp.predict(test_x, batch_size=batch_size) / len(self.model_list)
                if withBoosting:
                    # iteration on all boosters
                    prob_concat = [result_dic["prob"]]
                    for booster in self.boosters.values():
                        # iteration on each models (best iteration parameter is not set)
                        booster_prob = np.zeros((test_x.shape[0], 1))
                        for i in booster["model_list"]:
                            booster_prob += i.predict_proba(test_x)[...,1,np.newaxis] / len(booster["model_list"])
                        prob_concat.append(booster_prob)
                    prob_concat = np.concatenate(prob_concat, axis=1)
                    result_dic["prob"] = self.meta_learner.predict_proba(prob_concat)[:,1,np.newaxis]

                result_dic["pred"] = [1 if i >= cut_off else 0 for i in result_dic["prob"][:,0]]
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"]),
                                                 "F1": metrics.f1_score(test_y, result_dic["pred"]),
                                                 "ROC_AUC": metrics.roc_auc_score(test_y, result_dic["prob"])}
            else:
                result_dic["prob"] = np.zeros((test_x.shape[0], len(class_levels)))
                for model_mlp in self.model_list:
                    result_dic["prob"] += model_mlp.predict(test_x, batch_size=batch_size) / len(self.model_list)
                if withBoosting:
                    prob_concat = [result_dic["prob"][..., np.newaxis]]
                    for booster in self.boosters.values():
                        # iteration on each models (best iteration parameter is not set)
                        booster_prob = np.zeros((test_x.shape[0], len(class_levels)))
                        for i in booster["model_list"]:
                            booster_prob += i.predict_proba(test_x) / len(booster["model_list"])
                        prob_concat.append(booster_prob[..., np.newaxis])
                    prob_concat = np.concatenate(prob_concat, axis=1).mean(axis=1)[..., np.newaxis]
                    result_dic["prob"] = self.meta_learner.predict_proba(prob_concat)

                result_dic["pred"] = result_dic["prob"].argmax(axis=1)
                if test_y is not None:
                    result_dic["performance"] = {"Logloss": metrics.log_loss(test_y, result_dic["prob"]),
                                                 "Accuracy": metrics.accuracy_score(test_y, result_dic["pred"])}

        return result_dic
    def mlpBoosting(self, train_x, train_y, val_x, val_y, test_x, kIdx, kfolds):
        # 1. Save fitted booster
        # 2. Predict k-folds and validation data

        for booster_key, booster_obj in self.boosters.items():
            if booster_key == "XGB_GBT":
                if targetType == "numeric":
                    if booster_obj["booster"] is None:
                        booster_kfold = KFold(5, shuffle=True, random_state=9000)
                        booster_obj["model_list"].append(doXGB(train_x, train_y,
                                                               val_x, val_y,
                                                               kfolds=booster_kfold,
                                                               subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                               model_export=True)["model"])
                    else:
                        booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                    booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                    booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                      eval_metric="rmse", verbose=False,
                                                      early_stopping_rounds=self.earlyStopping)

                    booster_obj["ensemble_pred"][kIdx] += \
                        booster_obj["model_list"][-1].predict(val_x)[...,np.newaxis]
                    print(booster_key + " FOLD Score :", metrics.mean_squared_error(val_y, booster_obj["ensemble_pred"][kIdx], squared=False))
                    print("Trees --->", booster_obj["model_list"][-1].best_iteration)
                    print()
                    booster_obj["pred"] += \
                        booster_obj["model_list"][-1].predict(test_x)[...,np.newaxis] / kfolds
                else:
                    if targetTask == "binary":

                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doXGB(train_x, train_y,
                                                                   val_x, val_y,
                                                                   kfolds=booster_kfold,
                                                                   subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                          eval_metric="logloss", verbose=False,
                                                          early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)[..., 1, np.newaxis]
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        print("Best iteration --->", booster_obj["model_list"][-1].best_iteration)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x)[..., 1, np.newaxis] / kfolds
                    else:
                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doXGB(train_x, train_y,
                                                                   val_x, val_y,
                                                                   kfolds=booster_kfold,
                                                                   subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                          eval_metric="mlogloss", verbose=False,
                                                          early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        print("Best iteration --->", booster_obj["model_list"][-1].best_iteration)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x) / kfolds
            elif booster_key == "LGB_GOSS":
                if targetType == "numeric":

                    if booster_obj["booster"] is None:
                        booster_kfold = KFold(5, shuffle=True, random_state=9000)
                        booster_obj["model_list"].append(doLGB(train_x, train_y,
                                                               val_x, val_y,
                                                               boostingType="goss", kfolds=booster_kfold,
                                                               subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                               model_export=True)["model"])
                    else:
                        booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                    booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                    booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                      categorical_feature=None, eval_metric="rmse",
                                                      verbose=False, early_stopping_rounds=self.earlyStopping)

                    booster_obj["ensemble_pred"][kIdx] += \
                        booster_obj["model_list"][-1].predict(val_x)[...,np.newaxis]
                    print(booster_key + " FOLD Score :", metrics.mean_squared_error(val_y, booster_obj["ensemble_pred"][kIdx], squared=False))
                    # print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                    print()
                    booster_obj["pred"] += \
                        booster_obj["model_list"][-1].predict(test_x)[...,np.newaxis] / kfolds
                else:
                    if targetTask == "binary":
                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doLGB(train_x, train_y,
                                                                   val_x, val_y,
                                                                   boostingType="goss", kfolds=booster_kfold,
                                                                   subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                          categorical_feature=None, eval_metric="binary_logloss",
                                                          verbose=False, early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)[..., 1, np.newaxis]
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        # print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x)[..., 1, np.newaxis] / kfolds
                    else:
                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doLGB(train_x, train_y,
                                                                   val_x, val_y,
                                                                   boostingType="goss", kfolds=booster_kfold,
                                                                   subsampleSeq=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(n_estimators=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, eval_set=[(val_x, val_y)],
                                                          categorical_feature=None, eval_metric="multi_logloss",
                                                          verbose=False, early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        # print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x) / kfolds
            elif booster_key == "CAT_GBM":
                if targetType == "numeric":

                    if booster_obj["booster"] is None:
                        booster_kfold = KFold(5, shuffle=True, random_state=9000)
                        booster_obj["model_list"].append(doCAT(train_x, train_y,
                                                               val_x, val_y,
                                                               boostingType="Plain", kfolds=booster_kfold,
                                                               random_strength=[0.8], colsampleSeq=[0.8],
                                                               model_export=True)["model"])
                    else:
                        booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                    booster_obj["model_list"][-1].set_params(iterations=self.ntrees)
                    booster_obj["model_list"][-1].fit(train_x, train_y, cat_features=None,
                                                      use_best_model=True,
                                                      eval_set=[(val_x, val_y)],
                                                      early_stopping_rounds=self.earlyStopping)

                    booster_obj["ensemble_pred"][kIdx] += \
                        booster_obj["model_list"][-1].predict(val_x)[...,np.newaxis]
                    print(booster_key + " FOLD Score :", metrics.mean_squared_error(val_y, booster_obj["ensemble_pred"][kIdx], squared=False))
                    print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                    print()
                    booster_obj["pred"] += \
                        booster_obj["model_list"][-1].predict(test_x)[...,np.newaxis] / kfolds
                else:
                    if targetTask == "binary":
                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doCAT(train_x, train_y,
                                                                   val_x, val_y,
                                                                   boostingType="Plain", kfolds=booster_kfold,
                                                                   random_strength=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(iterations=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, cat_features=None,
                                                          use_best_model=True,
                                                          eval_set=[(val_x, val_y)],
                                                          early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)[..., 1, np.newaxis]
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x)[..., 1, np.newaxis] / kfolds
                    else:
                        if booster_obj["booster"] is None:
                            booster_kfold = StratifiedKFold(5, shuffle=True, random_state=9000)
                            booster_obj["model_list"].append(doCAT(train_x, train_y,
                                                                   val_x, val_y,
                                                                   boostingType="Plain", kfolds=booster_kfold,
                                                                   random_strength=[0.8], colsampleSeq=[0.8],
                                                                   model_export=True)["model"])
                        else:
                            booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))

                        booster_obj["model_list"][-1].set_params(iterations=self.ntrees)
                        booster_obj["model_list"][-1].fit(train_x, train_y, cat_features=None,
                                                          use_best_model=True,
                                                          eval_set=[(val_x, val_y)],
                                                          early_stopping_rounds=self.earlyStopping)

                        booster_obj["ensemble_prob"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)
                        print(booster_key + " FOLD Score :", metrics.log_loss(val_y, booster_obj["ensemble_prob"][kIdx]))
                        print("Best iteration --->", booster_obj["model_list"][-1].best_iteration_)
                        print()
                        booster_obj["prob"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x) / kfolds
            else:
                print("Unknown Booster (default fitting) :", booster_key)
                booster_obj["model_list"].append(copy.deepcopy(booster_obj["booster"]))
                booster_obj["model_list"][-1].fit(train_x, train_y)
                if targetType == "numeric":
                    booster_obj["ensemble_pred"][kIdx] += \
                        booster_obj["model_list"][-1].predict(val_x)
                    booster_obj["pred"] += \
                        booster_obj["model_list"][-1].predict(test_x) / kfolds
                else:
                    if targetTask == "binary":
                        booster_obj["ensemble_pred"][kIdx] += \
                            booster_obj["model_list"][-1].predict_proba(val_x)[..., np.newaxis]
                    else:
                        booster_obj["pred"] += \
                            booster_obj["model_list"][-1].predict_proba(test_x) / kfolds
    def clear(self):
        self.ensemble_pred = None
        self.pred = None
        self.predWB = None
        self.prob = None
        self.probWB = None
        self.mlp_finalscore = None
        self.performance = {}
        self.performanceWB = {}
        self.model_list = []
        self.boosters["model_list"] = []
        self.running_time = 0

# input_epochs = 200
# input_batch_size = 4
#
# mlpWB_boosters = {"XGB_GBT": {"booster": None},
#                   "LGB_GOSS": {"booster": None},
#                   "CAT_GBM": {"booster": None}}
# # mlpWB_boosters = {"XGB_GBT": {"booster": xgb.XGBClassifier(booster="gbtree", objective="binary:logistic", colsample_bytree=0.8,
# #                                                            n_estimators=mlpWB_ntrees, random_state=1,
# #                                                            max_depth=result_val["XGB_GBT"]["best_params"]["max_depth"],
# #                                                            subsample=result_val["XGB_GBT"]["best_params"]["subsample"],
# #                                                            reg_lambda=result_val["XGB_GBT"]["best_params"]["reg_lambda"],
# #                                                            min_child_weight=result_val["XGB_GBT"]["best_params"]["min_child_weight"],
# #                                                            gamma=result_val["XGB_GBT"]["best_params"]["gamma"],
# #                                                            learning_rate=5e-3, n_jobs=multiprocessing.cpu_count(), verbosity=0)},
# #                   "LGB_GOSS": {"booster": lgb.LGBMClassifier(boosting_type="goss", objective="binary", colsample_bytree=0.8,
# #                                                              n_estimators=mlpWB_ntrees, random_state=2,
# #                                                              num_leaves=result_val["LGB_GOSS"]["best_params"]["num_leaves"],
# #                                                              subsample=result_val["LGB_GOSS"]["best_params"]["subsample"],
# #                                                              reg_lambda=result_val["LGB_GOSS"]["best_params"]["reg_lambda"],
# #                                                              min_child_weight=result_val["LGB_GOSS"]["best_params"]["min_child_weight"],
# #                                                              min_child_samples=result_val["LGB_GOSS"]["best_params"]["min_child_samples"],
# #                                                              min_split_gain=result_val["LGB_GOSS"]["best_params"]["min_split_gain"],
# #                                                              learning_rate=5e-3, n_jobs=multiprocessing.cpu_count(), silent=True)},
# #                   "CAT_GBM": {"booster": cat.CatBoostClassifier(boosting_type="Plain", loss_function="Logloss", rsm=0.8,
# #                                                                 n_estimators=mlpWB_ntrees, random_state=3,
# #                                                                 max_depth=result_val["CAT_GBM"]["best_params"]["max_depth"],
# #                                                                 bagging_temperature=result_val["CAT_GBM"]["best_params"]["bagging_temperature"],
# #                                                                 l2_leaf_reg=result_val["CAT_GBM"]["best_params"]["l2_leaf_reg"],
# #                                                                 random_strength=result_val["CAT_GBM"]["best_params"]["random_strength"],
# #                                                                 learning_rate=5e-2, thread_count=multiprocessing.cpu_count(), logging_level="Silent")}}
#
# # mlpWB_metaLearner = lm.ElasticNetCV(cv=10, max_iter=1000, alphas=np.linspace(1e-3, 1e+3, 100).tolist(), random_state=4,
# #                                     l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])
# mlpWB_metaLearner = lm.LogisticRegressionCV(penalty="elasticnet", solver="saga", multi_class="ovr", cv=kfolds_spliter, max_iter=1000, random_state=4,
#                                             Cs=np.linspace(1e-3, 1e+3, 100).tolist(),
#                                             l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])
#
# # kfoldsMLP_ResNet_V1
# result_val["kfoldsMLP_ResNet_V1"] = {}
# result_test["kfoldsMLP_ResNet_V1"] = {}
#
# input_hiddenLayers = 64
# # input_hiddenLayers = result_val["MLP_ResNet_V1"]["best_params"]["hiddenLayers"]
#
# result_val["kfoldsMLP_ResNet_V1"]["model"] = kfoldsMLP(nCols=ds["full_x_oh"].shape[1],
#                                                        boosters=mlpWB_boosters, meta_learner=mlpWB_metaLearner,
#                                                        ntrees=5000, earlyStopping=int(5000*0.2), mlpName="MLP_ResNet_V1",
#                                                        hiddenLayers=input_hiddenLayers, dropoutRate=0.2)
#
# result_val["kfoldsMLP_ResNet_V1"]["model"].fit_predict(ds["full_x_oh"], ds["full_y"],
#                                                        ds["test_x_oh"], None,
#                                                        epochs=input_epochs, batch_size=input_batch_size,
#                                                        kfolds=5, stratify=ds["full_y"],
#                                                        seed=7373, withBoosting=True, model_export=True)
#
# result_val["kfoldsMLP_ResNet_V1"]["performance"] = result_val["kfoldsMLP_ResNet_V1"]["model"].performanceWB["Total"]
# print(result_val["kfoldsMLP_ResNet_V1"]["performance"])
#
# result_test["kfoldsMLP_ResNet_V1"] = result_val["kfoldsMLP_ResNet_V1"]["model"].predict(ds["test_x_oh"], None, batch_size=input_batch_size)
# print(result_test["kfoldsMLP_ResNet_V1"]["pred"][:10])
#
#
#
# # # save obejcts
# # easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# # easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")
#
#
# # display the performance
# for k, v in result_val.items():
#     if v is not None:
#         if "performance" in v.keys():
#             print(k, "--->", v["performance"])



model_names = ["Linear", "ElasticNet", "KNN", "XGB_GBT",
               "LGB_RF", "LGB_GOSS", "ARIMA", "MLP_LSTM_V1"]
# ===== Automation Predict =====
# validation data evaluation
fit_runningtime = time()
# 데이터를 저장할 변수 설정
total_perf = None
for stock_name, stock_data in stock_dic.items():
    stock_data["perf_list"] = dict.fromkeys(model_names)
    stock_data["pred_list"] = dict.fromkeys(model_names)
    total_perf = dict.fromkeys(model_names)
    for i in model_names:
        stock_data["perf_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        stock_data["pred_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        total_perf[i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        for j in total_perf[i].keys():
            total_perf[i][j] = series(0, index=["MAE", "MAPE", "NMAE", "RMSE", "NRMSE", "R2", "Running_Time"])


target_timegap = 5
seqLength = 5
eval_from = "20210910"
for time_ngap in range(1,target_timegap+1):
    print(F"=== Target on N+{time_ngap} ===")
    for stock_name, stock_data in stock_dic.items():

        # remove date
        full_x = stock_data["df"][:-time_ngap]
        full_y = stock_data["target_list"][time_ngap][:-time_ngap]
        tmp_date = full_x["date"]
        arima_target = stock_data["target_list"][0]
        arima_date = stock_data["df"]["date"]

        # # validation 1
        # train_x = full_x[tmp_date <= datetime(2021, 9, 2)]
        # train_y = full_y[tmp_date <= datetime(2021, 9, 2)]
        # train_x_lstm = full_x[tmp_date <= datetime(2021, 9, 2)]
        # train_y_lstm = full_y[tmp_date <= datetime(2021, 9, 2)][seqLength - 1:]
        # arima_train = arima_target[arima_date <= datetime(2021, 9, 3)]
        #
        # val_x = full_x[tmp_date == datetime(2021, 9, 3)]
        # val_y = full_y[tmp_date == datetime(2021, 9, 3)]
        # val_x_lstm = full_x[(which(tmp_date == datetime(2021, 9, 3))-seqLength+1): \
        #                     (which(tmp_date == datetime(2021, 9, 3))+target_timegap+1)][:seqLength]
        # val_y_lstm = full_y[(which(tmp_date == datetime(2021, 9, 3))-seqLength+1): \
        #                     (which(tmp_date == datetime(2021, 9, 3))+target_timegap+1)][seqLength-1:seqLength]
        # validation 2
        train_x = full_x[tmp_date <= datetime(2021, 9, 9)]
        train_y = full_y[tmp_date <= datetime(2021, 9, 9)]
        train_x_lstm = full_x[tmp_date <= datetime(2021, 9, 9)]
        train_y_lstm = full_y[tmp_date <= datetime(2021, 9, 9)][seqLength - 1:]
        arima_train = arima_target[arima_date <= datetime(2021, 9, 10)]

        val_x = full_x[tmp_date == datetime(2021, 9, 10)]
        val_y = full_y[tmp_date == datetime(2021, 9, 10)]
        val_x_lstm = full_x[(which(tmp_date == datetime(2021, 9, 10)) - seqLength + 1): \
                            (which(tmp_date == datetime(2021, 9, 10)) + target_timegap + 1)][:seqLength]
        val_y_lstm = full_y[(which(tmp_date == datetime(2021, 9, 10)) - seqLength + 1): \
                            (which(tmp_date == datetime(2021, 9, 10)) + target_timegap + 1)][seqLength - 1:seqLength]


        test_x = full_x[tmp_date == datetime(2021, 9, 24)]
        test_y = full_y[tmp_date == datetime(2021, 9, 24)]
        test_x_lstm = full_x[(which(tmp_date == datetime(2021, 9, 24))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 24))+target_timegap+1)][:seqLength]
        test_y_lstm = full_y[(which(tmp_date == datetime(2021, 9, 24))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 24))+target_timegap+1)][seqLength-1:seqLength]

        arima_full = arima_target[arima_date <= datetime(2021, 9, 24)]
        full_x_lstm = full_x[tmp_date <= datetime(2021, 9, 23)]
        full_y_lstm = full_y[tmp_date <= datetime(2021, 9, 23)][seqLength-1:]
        full_x = full_x[tmp_date <= datetime(2021, 9, 23)]
        full_y = full_y[tmp_date <= datetime(2021, 9, 23)]

        full_x.drop("date", axis=1, inplace=True)
        full_x_lstm.drop("date", axis=1, inplace=True)
        train_x.drop("date", axis=1, inplace=True)
        train_x_lstm.drop("date", axis=1, inplace=True)
        val_x.drop("date", axis=1, inplace=True)
        val_x_lstm.drop("date", axis=1, inplace=True)
        test_x.drop("date", axis=1, inplace=True)
        test_x_lstm.drop("date", axis=1, inplace=True)

        # <선형회귀>
        tmp_runtime = time()
        print("Linear Regression on", stock_name)
        # evaludation on validation set
        model = doLinear(train_x, train_y, val_x, val_y)
        print(model["performance"])
        stock_data["perf_list"]["Linear"][time_ngap] = model["performance"]
        tmp_perf = series(model["performance"])

        # prediction on test set
        model = doLinear(full_x, full_y, test_x, test_y)
        stock_data["pred_list"]["Linear"][time_ngap] = model["pred"]
        tmp_runtime = time() - tmp_runtime
        total_perf["Linear"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <엘라스틱넷>
        tmp_runtime = time()
        print("ElasticNet on", stock_name)
        # evaludation on validation set
        model = doElasticNet(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
        print(model["performance"])
        stock_data["perf_list"]["ElasticNet"][time_ngap] = model["performance"]
        tmp_perf = series(model["performance"])
        # prediction on test set
        model = doElasticNet(full_x, full_y, test_x, test_y, kfolds=kfolds_spliter, tuningMode=False,
                             alpha=model["best_params"]["alpha"], l1_ratio=model["best_params"]["l1_ratio"])
        stock_data["pred_list"]["ElasticNet"][time_ngap] = model["pred"]
        tmp_runtime = time() - tmp_runtime
        total_perf["ElasticNet"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <KNN>
        tmp_runtime = time()
        print("KNN on", stock_name)
        # evaludation on validation set
        model = doKNN(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter)
        print(model["performance"])
        stock_data["perf_list"]["KNN"][time_ngap] = model["performance"]
        tmp_perf = series(model["performance"])
        # prediction on test set
        model = doKNN(full_x, full_y, test_x, test_y, kfolds=kfolds_spliter)
        stock_data["pred_list"]["KNN"][time_ngap] = model["pred"]
        tmp_runtime = time() - tmp_runtime
        total_perf["KNN"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <XGBoost>
        tmp_runtime = time()
        print("XGB_GBT on", stock_name)
        # evaludation on validation set
        model = doXGB(train_x, train_y, val_x, val_y, kfolds=kfolds_spliter,
                      depthSeq=[6], subsampleSeq=[0.8], colsampleSeq=[1.0], gammaSeq=[0.0])
        print(model["performance"])
        stock_data["perf_list"]["XGB_GBT"][time_ngap] = model["performance"]
        tmp_perf = series(model["performance"])
        print(model["best_params"])
        # prediction on test set
        model = doXGB(full_x, full_y, test_x, test_y, tuningMode=False,
                      ntrees=model["best_params"]["best_trees"],
                      depthSeq=model["best_params"]["max_depth"],
                      mcwSeq=model["best_params"]["min_child_weight"],
                      l2Seq=model["best_params"]["reg_lambda"],
                      gammaSeq=model["best_params"]["gamma"],
                      subsampleSeq=model["best_params"]["subsample"],
                      colsampleSeq=model["best_params"]["colsample_bytree"])
        stock_data["pred_list"]["XGB_GBT"][time_ngap] = model["pred"]
        tmp_runtime = time() - tmp_runtime
        total_perf["XGB_GBT"][time_ngap] += tmp_perf.append(series({"Running_Time": tmp_runtime}))

        # <LightGBM 랜덤포레스트>
        tmp_runtime = time()
        # GridSearchCV의 param_grid 설정
        print("LGB_RF on", stock_name)
        params = {
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [2**i-1 for i in [4, 6, 8]],
            'n_estimators': [300, 400, 500],
            "subsample": [0.6, 0.8]
        }

        model = lgb.LGBMRegressor(boosting_type='rf', objective="regression", subsample_freq=2,
                                  n_jobs=None, random_state=321)
        grid = GridTuner(estimator=model, param_grid=params, n_jobs=multiprocessing.cpu_count(),
                         refit=False, cv=kfolds_spliter)
        grid.fit(train_x, train_y)

        # 하이퍼 파라미터 1,2,3차 수정
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 300, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        model = lgb.LGBMRegressor(boosting_type='rf', objective="regression", subsample_freq=2,
                                  n_estimators=grid.best_params_["n_estimators"],
                                  num_leaves=grid.best_params_["num_leaves"],
                                  subsample=grid.best_params_["subsample"],
                                  learning_rate=grid.best_params_["learning_rate"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=100, eval_metric='rmse', verbose=0)
        pred = model.predict(val_x)
        print(grid.best_params_)
        print("best iteration --->", model.best_iteration_)

        # recode performance
        tmp_mae = metrics.mean_absolute_error(val_y, pred)
        tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
        model_perf = {"MAE": tmp_mae,
                      "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                      "NMAE": tmp_mae / val_y.abs().mean(),
                      "RMSE": tmp_rmse,
                      "NRMSE": tmp_rmse / val_y.abs().mean(),
                      "R2": metrics.r2_score(val_y, pred)}
        print(tmp_perf)
        stock_data["perf_list"]["LGB_RF"][time_ngap] = model_perf

        # prediction on test data
        model = lgb.LGBMRegressor(boosting_type='rf', objective="regression", subsample_freq=2,
                                  n_estimators=model.best_iteration_,
                                  num_leaves=grid.best_params_["num_leaves"],
                                  subsample=grid.best_params_["subsample"],
                                  learning_rate=grid.best_params_["learning_rate"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(full_x, full_y, verbose=0)
        pred = model.predict(test_x)
        stock_data["pred_list"]["LGB_RF"][time_ngap] = pred
        # recode running time
        tmp_runtime = time() - tmp_runtime
        print(tmp_runtime)
        total_perf["LGB_RF"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))



        # <LightGBM Gradient-based One-Side Sampling>
        tmp_runtime = time()
        # GridSearchCV의 param_grid 설정
        print("LGB_GOSS on", stock_name)
        params = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [1000, 2000, 3000],
            'num_leaves': [2 ** i - 1 for i in [4, 6, 8]]
        }

        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8, n_jobs=None, random_state=321)
        grid = GridTuner(estimator=model, param_grid=params, n_jobs=multiprocessing.cpu_count(),
                         refit=False, cv=kfolds_spliter)
        grid.fit(train_x, train_y)

        # 하이퍼 파라미터 1,2,3차 수정
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 300, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression",  subsample=0.8,
                                  n_estimators=grid.best_params_["n_estimators"],
                                  num_leaves=grid.best_params_["num_leaves"],
                                  learning_rate=grid.best_params_["learning_rate"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=500, eval_metric='rmse', verbose=0)
        pred = model.predict(val_x)
        print(grid.best_params_)
        print("best iteration --->", model.best_iteration_)

        # recode performance
        tmp_mae = metrics.mean_absolute_error(val_y, pred)
        tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
        model_perf = {"MAE": tmp_mae,
                      "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                      "NMAE": tmp_mae / val_y.abs().mean(),
                      "RMSE": tmp_rmse,
                      "NRMSE": tmp_rmse / val_y.abs().mean(),
                      "R2": metrics.r2_score(val_y, pred)}
        print(tmp_perf)
        stock_data["perf_list"]["LGB_GOSS"][time_ngap] = model_perf

        # prediction on test data
        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                  n_estimators=model.best_iteration_,
                                  num_leaves=grid.best_params_["num_leaves"],
                                  learning_rate=grid.best_params_["learning_rate"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(full_x, full_y, verbose=0)
        pred = model.predict(test_x)
        stock_data["pred_list"]["LGB_GOSS"][time_ngap] = pred
        # recode running time
        tmp_runtime = time() - tmp_runtime
        print(tmp_runtime)
        total_perf["LGB_GOSS"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))

        # <ARIMA>
        tmp_runtime = time()
        print("ARIMA on", stock_name)
        # order=(p: Auto regressive, q: Difference, d: Moving average)
        # 일반적 하이퍼파라미터 공식
        # 1. p + q < 2
        # 2. p * q = 0
        # 근거 : 실제로 대부분의 시계열 자료에서는 하나의 경향만을 강하게 띄기 때문 (p 또는 q 둘중 하나는 0)
        model = ARIMA(arima_train, order=(1, 2, 0))
        model_fit = model.fit()
        pred = array([model_fit.forecast(time_ngap).iloc[-1]])
        tmp_mae = metrics.mean_absolute_error(val_y, pred)
        tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
        tmp_perf = {"MAE": tmp_mae,
                      "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                      "NMAE": tmp_mae / val_y.abs().mean(),
                      "RMSE": tmp_rmse,
                      "NRMSE": tmp_rmse / val_y.abs().mean(),
                      "R2": metrics.r2_score(val_y, pred)}
        print(tmp_perf)
        stock_data["perf_list"]["ARIMA"][time_ngap] = tmp_perf

        # prediction on test data
        model = ARIMA(arima_full, order=(1, 2, 0))
        model_fit = model.fit()

        stock_data["pred_list"]["ARIMA"][time_ngap] = array([model_fit.forecast(time_ngap).iloc[-1]])
        # recode running time
        tmp_runtime = time() - tmp_runtime
        total_perf["ARIMA"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))

        # <LSTM>
        tmp_runtime = time()
        print("LSTM V1 on", stock_name)
        model = doMLP(train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, mlpName="MLP_LSTM_V1",
                      hiddenLayers=128, epochs=100, batch_size=2, seqLength=seqLength, model_export=True)
        print(model["performance"])
        stock_data["perf_list"]["MLP_LSTM_V1"][time_ngap] = model["performance"]
        tmp_perf = series(model["performance"])

        model = doMLP(full_x_lstm, full_y_lstm, test_x_lstm, test_y_lstm,
                      seqLength=seqLength, preTrained=model["model"])
        stock_data["pred_list"]["MLP_LSTM_V1"][time_ngap] = model["pred"]
        tmp_runtime = time() - tmp_runtime
        total_perf["MLP_LSTM_V1"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))

    for i in model_names:
        total_perf[i][time_ngap] /= len(stock_dic.keys())
fit_runningtime = time() - fit_runningtime


# prediction value check
print(stock_dic["삼성전자"]["pred_list"])
print(fit_runningtime)
stock_dic["삼성전자"]["df"]


# 성능평가 테이블 생성
perf_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
runningtime_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
for i in list(total_perf.keys()):
    if array(list(total_perf[i].values())).sum() == 0:
        pass
    else:
        perf_table.loc[i] = dataframe(total_perf[i]).loc["NMAE"].values
        runningtime_table.loc[i] = dataframe(total_perf[i]).loc["Running_Time"].values

# NMAE = MAPE
perf_table = perf_table.iloc[:,:target_timegap]
perf_table = perf_table * 100
perf_table.loc["best_model"] = perf_table.min(axis=0)
perf_table["avg"] = perf_table.iloc[:,:5].mean(axis=1)
perf_table["std"] = perf_table.iloc[:,:5].std(axis=1)
perf_table["running_time"] = runningtime_table.mean(axis=1).append(series({"best_model": -1}))
print(perf_table)

smoothing_alpha = 1
effic_table = (1/(perf_table.iloc[:,:target_timegap] + 1e-1 * smoothing_alpha))
effic_table = effic_table.apply(lambda x: x/(perf_table["running_time"] + 1e-5 * smoothing_alpha))
effic_table = effic_table * 100
effic_table["avg"] = effic_table.iloc[:,:5].mean(axis=1)
effic_table["std"] = effic_table.iloc[:,:5].std(axis=1)
print(effic_table.mean(axis=1))

selected_features

# perf_table.to_csv("projects/dacon_stockprediction/eval_result/perf_" + feature_test_seed + "_" + eval_from + ".csv")
# effic_table.to_csv("projects/dacon_stockprediction/eval_result/effic_" + feature_test_seed + "_" + eval_from + ".csv")








# test data prediction

# best feature : feature test 3
# best model : LGB_GOSS

model_names = ["LGB_GOSS"]

fit_runningtime = time()
# 데이터를 저장할 변수 설정
total_perf = None
for stock_name, stock_data in stock_dic.items():
    stock_data["perf_list"] = dict.fromkeys(model_names)
    stock_data["pred_list"] = dict.fromkeys(model_names)
    stock_data["model_list"] = dict.fromkeys(model_names)
    stock_data["pred_df"] = dataframe(columns=[1, 2, 3, 4, 5])
    total_perf = dict.fromkeys(model_names)
    for i in model_names:
        stock_data["perf_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        stock_data["pred_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        stock_data["model_list"][i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        total_perf[i] = dict.fromkeys([1, 2, 3, 4, 5], 0)
        for j in total_perf[i].keys():
            total_perf[i][j] = series(0, index=["MAE", "MAPE", "NMAE", "RMSE", "NRMSE", "R2", "Running_Time"])


target_timegap = 5
seqLength = 5
for time_ngap in range(1,target_timegap+1):
    print(F"=== Target on N+{time_ngap} ===")
    for stock_name, stock_data in stock_dic.items():

        # remove date
        full_x = stock_data["df"][:-time_ngap]
        full_y = stock_data["target_list"][time_ngap][:-time_ngap]
        tmp_date = full_x["date"]
        arima_target = stock_data["target_list"][0]
        arima_date = stock_data["df"]["date"]

        # validation for test dataset
        train_x = full_x[tmp_date <= datetime(2021, 9, 16)]
        train_y = full_y[tmp_date <= datetime(2021, 9, 16)]
        train_x_lstm = full_x[tmp_date <= datetime(2021, 9, 16)]
        train_y_lstm = full_y[tmp_date <= datetime(2021, 9, 16)][seqLength-1:]
        arima_train = arima_target[arima_date <= datetime(2021, 9, 17)]

        val_x = full_x[tmp_date == datetime(2021, 9, 17)]
        val_y = full_y[tmp_date == datetime(2021, 9, 17)]
        val_x_lstm = full_x[(which(tmp_date == datetime(2021, 9, 17))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 17))+target_timegap+1)][:seqLength]
        val_y_lstm = full_y[(which(tmp_date == datetime(2021, 9, 17))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 17))+target_timegap+1)][seqLength-1:seqLength]

        test_x = full_x[tmp_date == datetime(2021, 9, 24)]
        test_y = full_y[tmp_date == datetime(2021, 9, 24)]
        test_x_lstm = full_x[(which(tmp_date == datetime(2021, 9, 24))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 24))+target_timegap+1)][:seqLength]
        test_y_lstm = full_y[(which(tmp_date == datetime(2021, 9, 24))-seqLength+1): \
                            (which(tmp_date == datetime(2021, 9, 24))+target_timegap+1)][seqLength-1:seqLength]

        arima_full = arima_target[arima_date <= datetime(2021, 9, 24)]
        full_x_lstm = full_x[tmp_date <= datetime(2021, 9, 23)]
        full_y_lstm = full_y[tmp_date <= datetime(2021, 9, 23)][seqLength-1:]
        full_x = full_x[tmp_date <= datetime(2021, 9, 23)]
        full_y = full_y[tmp_date <= datetime(2021, 9, 23)]

        full_x.drop("date", axis=1, inplace=True)
        full_x_lstm.drop("date", axis=1, inplace=True)
        train_x.drop("date", axis=1, inplace=True)
        train_x_lstm.drop("date", axis=1, inplace=True)
        val_x.drop("date", axis=1, inplace=True)
        val_x_lstm.drop("date", axis=1, inplace=True)
        test_x.drop("date", axis=1, inplace=True)
        test_x_lstm.drop("date", axis=1, inplace=True)

        # <LightGBM Gradient-based One-Side Sampling>
        tmp_runtime = time()
        # GridSearchCV의 param_grid 설정
        print("LGB_GOSS on", stock_name)
        params = {
            'learning_rate': [5e-4],
            'n_estimators': [2000],
            'num_leaves': [2 ** i - 1 for i in [4, 6]],
            'reg_lambda': [0.1, 1.0, 5.0],
            'min_child_samples': [5, 10, 20]
        }
        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8, n_jobs=None, random_state=321)
        grid = GridTuner(estimator=model, param_grid=params, n_jobs=multiprocessing.cpu_count(),
                         refit=False, cv=kfolds_spliter)
        grid.fit(train_x, train_y)

        # 하이퍼 파라미터 1,2,3차 수정
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 300, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        # best_params_: {'learning_rate': 0.05, 'n_estimators': 400, 'num_leaves': 7}
        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression",  subsample=0.8,
                                  n_estimators=5000,
                                  num_leaves=grid.best_params_["num_leaves"],
                                  reg_lambda=grid.best_params_["reg_lambda"],
                                  min_child_samples=grid.best_params_["min_child_samples"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(train_x, train_y, eval_set=(val_x, val_y), early_stopping_rounds=500, eval_metric='rmse', verbose=0)
        pred = model.predict(val_x)
        print(grid.best_params_)
        print("best iteration --->", model.best_iteration_)

        # recode performance
        tmp_mae = metrics.mean_absolute_error(val_y, pred)
        tmp_rmse = metrics.mean_squared_error(val_y, pred, squared=False)
        model_perf = {"MAE": tmp_mae,
                      "MAPE": metrics.mean_absolute_percentage_error(val_y, pred),
                      "NMAE": tmp_mae / val_y.abs().mean(),
                      "RMSE": tmp_rmse,
                      "NRMSE": tmp_rmse / val_y.abs().mean(),
                      "R2": metrics.r2_score(val_y, pred)}
        print(model_perf)
        stock_data["perf_list"]["LGB_GOSS"][time_ngap] = model_perf

        # prediction on test data
        model = lgb.LGBMRegressor(boosting_type='goss', objective="regression", subsample=0.8,
                                  n_estimators=model.best_iteration_,
                                  num_leaves=grid.best_params_["num_leaves"],
                                  reg_lambda=grid.best_params_["reg_lambda"],
                                  min_child_samples=grid.best_params_["min_child_samples"],
                                  n_jobs=multiprocessing.cpu_count(), random_state=321)
        model.fit(full_x, full_y, verbose=0)
        stock_data["model_list"]["LGB_GOSS"][time_ngap] = model
        pred = model.predict(test_x)
        stock_data["pred_list"]["LGB_GOSS"][time_ngap] = pred

        # recode running time
        tmp_runtime = time() - tmp_runtime
        print(tmp_runtime)
        total_perf["LGB_GOSS"][time_ngap] += series(model_perf).append(series({"Running_Time": tmp_runtime}))

        # predict full dataset for visualization
        pred = model.predict(full_x)
        stock_data["pred_df"][time_ngap] = pred

        # # <LSTM>
        # tmp_runtime = time()
        # print("LSTM V1 on", stock_name)
        # model_mlp = doMLP(train_x_lstm, train_y_lstm, val_x_lstm, val_y_lstm, mlpName="MLP_LSTM_V1",
        #               hiddenLayers=128, epochs=100, batch_size=2, seqLength=seqLength, model_export=True)
        # stock_data["model_list"]["MLP_LSTM_V1"][time_ngap] = model_mlp["model"]
        # print(model_mlp["performance"])
        # stock_data["perf_list"]["MLP_LSTM_V1"][time_ngap] = model_mlp["performance"]
        # tmp_perf = series(model_mlp["performance"])
        #
        # model_mlp_test = doMLP(full_x_lstm, full_y_lstm, test_x_lstm, test_y_lstm,
        #                        seqLength=seqLength, preTrained=model_mlp["model"])
        # stock_data["pred_list"]["MLP_LSTM_V1"][time_ngap] = model_mlp_test["pred"]
        # tmp_runtime = time() - tmp_runtime
        # total_perf["MLP_LSTM_V1"][time_ngap] += series(tmp_perf).append(series({"Running_Time": tmp_runtime}))
        #
        # # predict full dataset for visualization
        # model_mlp_full = doMLP(full_x_lstm, full_y_lstm, full_x_lstm, full_y_lstm,
        #                        seqLength=seqLength, preTrained=model_mlp["model"])
        #
        # stock_data["pred_df"][time_ngap] = dd = [nan]*(seqLength-1) + list(model_mlp_full["pred"].flatten())
    for i in model_names:
        total_perf[i][time_ngap] /= len(stock_dic.keys())

fit_runningtime = time() - fit_runningtime



# prediction value check
print(stock_dic["삼성전자"]["pred_list"])
print(fit_runningtime)
stock_dic["삼성전자"]["df"]


# 성능평가 테이블 생성
perf_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
runningtime_table = dataframe(index=model_names, columns=["time_gap_" + str(i) for i in range(1,6)])
for i in list(total_perf.keys()):
    if array(list(total_perf[i].values())).sum() == 0:
        pass
    else:
        perf_table.loc[i] = dataframe(total_perf[i]).loc["NMAE"].values
        runningtime_table.loc[i] = dataframe(total_perf[i]).loc["Running_Time"].values

# NMAE = MAPE
perf_table = perf_table.iloc[:,:target_timegap]
perf_table = perf_table * 100
perf_table.loc["best_model"] = perf_table.min(axis=0)
perf_table["avg"] = perf_table.iloc[:,:5].mean(axis=1)
perf_table["std"] = perf_table.iloc[:,:5].std(axis=1)
perf_table["running_time"] = runningtime_table.mean(axis=1).append(series({"best_model": -1}))
print(perf_table)

smoothing_alpha = 1
effic_table = (1/(perf_table.iloc[:,:target_timegap] + 1e-1 * smoothing_alpha))
effic_table = effic_table.apply(lambda x: x/(perf_table["running_time"] + 1e-5 * smoothing_alpha))
effic_table = effic_table * 100
effic_table["avg"] = effic_table.iloc[:,:5].mean(axis=1)
effic_table["std"] = effic_table.iloc[:,:5].std(axis=1)
print(effic_table.mean(axis=1))



for time_ngap in range(1,target_timegap+1):
    print(F"=== Target on N+{time_ngap} ===")
    for stock_name, stock_data in stock_dic.items():
        stock_name
        stock_data["pred_df"]


test_pred = []
for i in stock_data["pred_list"]["LGB_GOSS"].values():
    test_pred.append(i[0])

submission = read_csv("projects/dacon_stockprediction/open_week4/sample_submission_week4.csv")
submission = submission.iloc[5:,:].reset_index(drop=True)

# test set prediction
# for i in submission.columns[1:]:
#     print(stock_dic[stock_list.index[stock_list == i][0]]["pred_list"])
#     submission[i] = [i.round()[0] for i in stock_dic[stock_list.index[stock_list == i][0]]["pred_list"]["LGB_GOSS"].values()]
#
# print(submission.isna().sum().sum())
# submission.to_csv("projects/dacon_stockprediction/submission/submission_lightgbm_goss.csv", index=False)

# test set evaluation
submission = read_csv("projects/dacon_stockprediction/submission/submission_lightgbm_goss.csv")

evalVec = []
for stock_code in submission.columns[1:]:
    actual = stock.get_market_ohlcv_by_date("20210927", "20211001", stock_code)["종가"]
    evalVec.append((np.abs(actual.values - submission[stock_code].values) / actual.values).mean())

print(np.mean(evalVec) * 100)





