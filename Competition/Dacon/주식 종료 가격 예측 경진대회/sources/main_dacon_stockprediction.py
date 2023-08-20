import os
import IPython
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
from numpy import array, nan, random as rnd, where as which
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm

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

# # ===== tensorflow =====
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras import layers
# from tensorflow.keras import activations
# from tensorflow.keras import optimizers
# from tensorflow.keras import metrics as tf_metrics
# from tensorflow.keras import callbacks as tf_callbacks
# from tqdm.keras import TqdmCallback
# from scikeras.wrappers import KerasClassifier, KerasRegressor
# import tensorflow_addons as tfa
# import keras_tuner as kt
# from keras_tuner import HyperModel

# ===== NLP =====
from selenium import webdriver
from konlpy.tag import Okt
from KnuSentiLex.knusl import KnuSL

# ===== task specific =====
import pykrx

# display setting
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
            naIdx[i] = list(which(array(x[i].isna()))[0])
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
def dispPerformance(result_dic, result_metrics):
    perf_table = dataframe(columns=result_metrics)
    for k, v in result_dic.items():
        perf_table = pd.concat([perf_table, v["performance"]], ignore_index=True, axis=0)
    print(perf_table)
    return perf_table
folder_path = "./projects/dacon_stockprediction/"

from pykrx import stock



# Get Stock List
path = 'projects/dacon_stockprediction/open_week4/'
list_name = 'Stock_List.csv'
sample_name = 'sample_submission_week4.csv'

# 종목 코드 로드
stock_list = read_csv(os.path.join(path, list_name))
stock_list['종목코드'] = stock_list['종목코드'].apply(lambda x: str(x).zfill(6))
stock_list



# Get Data & Modeling
# 분석할 date 변수 지정
start_date = '20201201'
end_date = '20211001'

start_weekday = pd.to_datetime(start_date).weekday()
max_weeknum = pd.to_datetime(end_date).strftime('%V')
business_days = pd.DataFrame(pd.date_range(start_date, end_date, freq='B'), columns=['Date'])

print(f'WEEKDAY of "start_date" : {start_weekday}')
print(f'NUM of WEEKS to "end_date" : {max_weeknum}')
print(f'HOW MANY "Business_days" : {business_days.shape}', )
print(business_days.head(20))

# raw features (5개)
# 주가, 거래량, 기관순매수, 외인순매수, 뉴스 기사(embedding)

# derived features (14개)
# 주가이평, 거래량이평, 기관순매수이평, 외인순매수이평, 뉴스 기사에 대한 긍부정점수, 요일, sin변환(5일), cos변환(5일)
# 산식 보조 지표
# 1. 주가 관련 지표 : Stochastic(20), RSI(20), 볼린저밴드(20)
# 2. 거래량 관련 지표 : OBV, VR(20)
# 3. 혼합지표 : MFI(주가 + 거래량)



# ===== raw data loading =====
# 한 종목코드에 대한 주가 정보를 로드
# 임의 선별
# 삼성전자
# NAVER
# 카카오

# 랜덤 선별
rnd.seed(48)
stock_list.iloc[rnd.randint(len(stock_list))]
# 금호석유
# 티움바이오
# 테크윙
# 제테마
# 주성엔지니어링
# 고바이오랩
# 고영

codes = ["삼성전자", "NAVER", "카카오", "금호석유", "티움바이오", "테크윙", "제테마", "주성엔지니어링", "고바이오랩", "고영"]
print(len(codes))

stock_list = stock_list.set_index("종목명")
stock_code = stock_list.loc[[""],'종목코드']
stock_df = stock.get_market_ohlcv_by_date(start_date, end_date, stock_code).reset_index()
investor_df = stock.get_market_trading_volume_by_date(start_date, end_date, stock_code)[["기관합계", "외국인합계"]].reset_index()
kospi_df = stock.get_index_ohlcv_by_date(start_date, end_date, "1001")[["종가"]].reset_index()

stock_df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
investor_df.columns = ["Date", "inst", "fore"]
kospi_df.columns = ["Date", "kospi"]
# 영업일과 주가 정보를 outer 조인
train_x = pd.merge(business_days, stock_df, how='left', on="Date")
train_x = pd.merge(train_x, investor_df, how='left', on="Date")
train_x = pd.merge(train_x, kospi_df, how='left', on="Date")
# 종가데이터에 생긴 na 값을 선형보간 및 정수로 반올림
train_x.iloc[:,1:] = train_x.iloc[:,1:].ffill(axis=0).round(0)

print(train_x.isna().sum())

# ===== feature engineering =====
# 요일 및 주차 파생변수 추가
train_x['weekday'] = train_x["Date"].apply(lambda x: x.weekday())
train_x['weeknum'] = train_x["Date"].apply(lambda x: week_of_month(x))
cat_vars = ["weekday", "weeknum"]

# 주기성 신호로 변환한 파생변수 추가 (이건 요일 특성을 잡아주는거랑 다를바가 없으니 다른 접근 필요)
# 차라리 해당 월에 몇번째 일 and 해당 년 몇번째 일인지

day_to_sec = 24 * 60 * 60
month_to_sec = 20 * day_to_sec

timestamp_s = train_x["Date"].apply(datetime.timestamp)
timestamp_freq = round((timestamp_s / month_to_sec).diff(20)[20],1)

train_x['dayofmonth_freq_sin'] = np.sin((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))
train_x['dayofmonth_freq_cos'] = np.cos((timestamp_s / month_to_sec) * ((2 * np.pi) / timestamp_freq))

# sns.lineplot(data=train_x['monthday_freq_sin'][:-1], color="g")
# ax2 = plt.twinx()
# sns.lineplot(data=train_x["Close"][1:], color="b", ax=ax2)
# train_x.head(30)
np.corrcoef(train_x['monthday_freq_sin'][:-1], train_x["Close"][1:])

# sns.lineplot(data=train_x['monthday_freq_cos'][:-1], color="g")
# ax2 = plt.twinx()
# sns.lineplot(data=train_x["Close"][1:], color="b", ax=ax2)
# train_x.head(30)
np.corrcoef(train_x['monthday_freq_cos'][:-1], train_x["Close"][1:])

day_to_sec = 24 * 60 * 60
weekday_to_sec = 5 * day_to_sec

timestamp_s = train_x["Date"].apply(datetime.timestamp)
timestamp_freq = round((timestamp_s / weekday_to_sec).diff(5)[5],1)

train_x['weekday_freq_sin'] = np.sin((timestamp_s / weekday_to_sec) * ((2 * np.pi) / timestamp_freq))
train_x['weekday_freq_cos'] = np.cos((timestamp_s / weekday_to_sec) * ((2 * np.pi) / timestamp_freq))

# sns.lineplot(data=train_x['weekday_freq_sin'][:-1], color="g")
# ax2 = plt.twinx()
# sns.lineplot(data=train_x["Close"][1:], color="b", ax=ax2)
# train_x.head(30)
np.corrcoef(train_x['weekday_freq_sin'][:-1], train_x["Close"][1:])

# sns.lineplot(data=train_x['weekday_freq_cos'][:-1], color="g")
# ax2 = plt.twinx()
# sns.lineplot(data=train_x["Close"][1:], color="b", ax=ax2)
# train_x.head(30)
np.corrcoef(train_x['weekday_freq_cos'][:-1], train_x["Close"][1:])
train_x.drop(['monthday_freq_cos', 'weekday_freq_sin', 'weekday_freq_cos'], axis=1, inplace=True)

# setting metrics days
metric_days = 14

# obv
obv = [0]
for i in range(1, len(train_x.Close)):
    if train_x.Close[i] > train_x.Close[i - 1]:
        obv.append(obv[-1] + train_x.Volume[i])
    elif train_x.Close[i] < train_x.Close[i - 1]:
        obv.append(obv[-1] - train_x.Volume[i])
    else:
        obv.append(obv[-1])
train_x['obv'] = obv
train_x['obv'][0] = nan
train_x['obv_ema'] = train_x['obv'].ewm(com=metric_days, min_periods=metric_days).mean()


# 매수/매도 타이밍 신호 찾는 함수
# 매수 신호: obv > obv_ema
# 매도 신호: obv < obv_ema
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

# train_x['obv_signal'] = getBreakthroughPoint(train_x, 'obv', 'obv_ema', 2)
train_x


# #OBV와 OBV_EMA 시각화
# plt.figure(figsize=(12,8))
# plt.plot(train_x['obv'], label='obv', color='orange')
# plt.plot(train_x['obv_ema'], label='obv_ema', color='purple')
# plt.legend(loc='upper right')
# plt.xticks(rotation=45)


# #매수/매도 신호 시각화
# plt.figure(figsize=(12,8))
# plt.scatter(train_x.index[train_x['obv_signal']=="buy"], train_x["Close"][train_x['obv_signal']=="buy"], color = 'green',
#             label = 'Buy Signal', marker = '^', alpha = 1)
# plt.scatter(train_x.index[train_x['obv_signal']=="sell"], train_x["Close"][train_x['obv_signal']=="sell"], color = 'red',
#             label = 'Sell Signal', marker = 'v', alpha = 1)
# # plt.plot(train_x['obv'], label = 'OBV', alpha = 0.35)
# # plt.plot(train_x['obv_ema'], label = 'OBV moving average', alpha = 0.35)
# plt.plot(train_x['Close'], label = 'Price', alpha = 0.35)
# plt.xticks(rotation=45)
# plt.title('Buy & Sell zone visualization', fontsize=15, fontweight="bold", pad=15)
# plt.xlabel('Date', fontsize = 14)
# plt.ylabel('Close Price', fontsize=14)
# plt.legend(loc='upper right')
# plt.show()


### stochastic 계산식
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

# 호출 방법
train_x[["fast_k", "fast_d", "slow_k", "slow_d"]] = stochastic(train_x, n=metric_days)[["fast_k", "fast_d", "slow_k", "slow_d"]]
# train_x['stochastic_signal'] = getBreakthroughPoint(train_x, 'fast_k', 'fast_d', 2)
train_x.head(20)

#MFI 지표 구하기
#MFI = 100 - (100/1+MFR)
#MFR = 14일간의 양의 MF/ 14일간의 음의 MF
#MF = 거래량 * (당일고가 + 당일저가 + 당일종가) / 3


train_x.tail(20)


#MF 컬럼 만들기
train_x["mf"] = train_x["Volume"] * ((train_x["High"]+train_x["Low"]+train_x["Close"]) / 3)
#양의 MF와 음의 MF 표기 컬럼 만들기
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

#14일간 양의 MF/ 14일간 음의 MF 계산하여 컬럼 만들기
mfr = []
for i in range(len(train_x['mf'])):
    if i < metric_days-1:
        mfr.append(nan)
    else:
        train_x_=train_x.iloc[(i-metric_days+1):i]
        a = sum(train_x_['mf'][train_x['p_n']=='p']) / sum(train_x_['mf'][train_x['p_n'] == 'n'])
        mfr.append(a)

train_x['mfr'] = mfr

# 최종 MFI 컬럼 만들기
train_x['mfi'] = 100 - (100/(1+train_x['mfr']))
# train_x["mfi_signal"] = train_x['mfi'].apply(lambda x: "buy" if x > 50 else "sell")
train_x.drop(["slow_k", "slow_d", "mf", "p_n", "mfr", "Open", "High", "Low"], inplace=True, axis=1)
train_x.head(20)

train_x["close_mv5"] = train_x["Close"].rolling(5, min_periods=5).mean()
train_x["close_mv10"] = train_x["Close"].rolling(10, min_periods=10).mean()
train_x["close_mv20"] = train_x["Close"].rolling(20, min_periods=20).mean()

train_x["volume_mv5"] = train_x["Volume"].rolling(5, min_periods=5).mean()
train_x["volume_mv10"] = train_x["Volume"].rolling(10, min_periods=10).mean()
train_x["volume_mv20"] = train_x["Volume"].rolling(20, min_periods=20).mean()

train_x["inst_mv5"] = train_x["inst"].rolling(5, min_periods=5).mean()
train_x["inst_mv10"] = train_x["inst"].rolling(10, min_periods=10).mean()
train_x["inst_mv20"] = train_x["inst"].rolling(20, min_periods=20).mean()

train_x["fore_mv5"] = train_x["fore"].rolling(5, min_periods=5).mean()
train_x["fore_mv10"] = train_x["fore"].rolling(10, min_periods=10).mean()
train_x["fore_mv20"] = train_x["fore"].rolling(20, min_periods=20).mean()

train_x["kospi_mv5"] = train_x["kospi"].rolling(5, min_periods=5).mean()
train_x["kospi_mv10"] = train_x["kospi"].rolling(10, min_periods=10).mean()
train_x["kospi_mv20"] = train_x["kospi"].rolling(20, min_periods=20).mean()

# 2021/1/4 이후 일자만 선택
train_x = train_x[train_x["Date"] >= datetime(2021, 1, 4)]
train_x = train_x.dropna()
train_x.reset_index(drop=True, inplace=True)

# create target list
target_list = []
target_list.append(train_x["Close"])
target_list.append(train_x["Close"].shift(-1))
target_list.append(train_x["Close"].shift(-2))
target_list.append(train_x["Close"].shift(-3))
target_list.append(train_x["Close"].shift(-4))
target_list.append(train_x["Close"].shift(-5))
for idx, value in enumerate(target_list[1:]):
    value.name = "close_shift" + str(idx+1)

train_x.columns = train_x.columns.str.lower()
train_x = pd.concat([train_x[["date"]], train_x.iloc[:,1:].sort_index(axis=1)], axis=1)
# bi_data = pd.concat([train_x, train_x["close"].shift(-1)], axis=1, ignore_index=True)[:-1]
# bi_data.columns = list(train_x.columns) + ["close_shift1"]
# bi_data.to_csv("projects/dacon_stockprediction/bi_data.csv", encoding="euc-kr", index=False)



# ===== visualization =====
# 상관관계 시각화
# fig, ax = plt.subplots(figsize=(12, 6))
# corr_obj = pd.concat([train_x[:-1], target_list[1][:-1]], axis=1).corr().round(3)
# sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True)
# # sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True)
# # plt.gcf().set_size_inches(16, 12)
# plt.show()
# # plt.savefig('projects/dacon_stockprediction/graphs/corr_heatmap.png', dpi=300)
# small_corr = corr_obj.index[corr_obj["close_shift1"].abs() < 0.1]
# small_corr = corr_obj["close_shift1"].abs().sum()
# plt.title('Correlation Visualization', fontsize=15, fontweight="bold", pad=15)
# train_x.head(20)

# # ===== scatter plot on numerical feature =====
# for i in train_x.columns:
#     if i == "Date" or i in cat_vars:
#         pass
#     else:
#         fig, ax = plt.subplots(figsize=(12, 6))
#         graph = sns.regplot(x=train_x[i][:-1], y=train_x["Close"][1:], color="green",
#                             scatter_kws={'s':15}, line_kws={"color": "orange"})
#         graph.set_title(i, fontsize=15, fontweight="bold", pad=15)
#         plt.show()
#         # plt.savefig('projects/dacon_stockprediction/graphs/' + i +".png", dpi=300)

# # ===== box plot on categorical feature =====
# for i in train_x.columns:
#     if i == "Date" or i not in cat_vars:
#         pass
#     else:
#         fig, ax = plt.subplots(figsize=(12,6))
#         graph = sns.boxplot(x=train_x[i][:-1], y=train_x["Close"][1:], palette=sns.hls_palette())
#         graph.set_title("Boxplot by " + i, fontsize=15, fontweight="bold", pad=15)
#         change_width(ax, 0.2)
#         plt.show()
#         # plt.savefig('projects/dacon_stockprediction/graphs/' + i +".png", dpi=300)



# 분산분석
from scipy.stats import f_oneway
tmp = pd.concat([train_x, target_list[1].to_frame()], axis=1, ignore_index=True)[1:-1]
tmp.columns = list(train_x.columns) + ["target"]

cat_list = tmp.groupby("weekday")["target"].apply(list)
# 귀무가설(H0) : 두 변수는 상관관계가 없다
# 대립가설(H1) : 두 변수는 상관관계가 있다
anova = f_oneway(*cat_list)
print(anova)

cat_list = tmp.groupby("weeknum")["target"].apply(list)
# 귀무가설(H0) : 두 변수는 상관관계가 없다
# 대립가설(H1) : 두 변수는 상관관계가 있다
anova = f_oneway(*cat_list)
print(anova)





# 영향력 적은 변수 제거 및 재시각화
# train_x.drop(["close_mv10", "close_mv20", "volume_mv5", "volume_mv10", "inst_mv5", "inst_mv20", "fore_mv5", "fore_mv10", "kospi_mv5", "kospi_mv10",
#               "monthday_freq_sin", "obv_ema", "fast_k", "mfi", "weekday", "weeknum"], axis=1, inplace=True)
# fig, ax = plt.subplots(figsize=(12, 6))
# corr_obj = pd.concat([train_x[:-1], target_list[1][1:]], axis=1).corr().round(3)
# sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True)
# # sns.heatmap(corr_obj, cmap="YlGnBu", linewidths=0.2, annot=True)
# # plt.gcf().set_size_inches(16, 12)
# plt.show()
# # plt.savefig('projects/dacon_stockprediction/graphs/corr_heatmap.png', dpi=300)
# small_corr = corr_obj.index[corr_obj["close_shift1"].abs() < 0.1]
# small_corr = corr_obj["close_shift1"].abs().sum()
# plt.title('Correlation Visualization', fontsize=15, fontweight="bold", pad=15)


onehot_encoder = MyOneHotEncoder()
train_x_oh = onehot_encoder.fit_transform(train_x, cat_vars)

print(train_x.info())
print(train_x_oh.info())



# dimension check
train_x.info()
train_x.head(10)
train_x_oh.head(10)

# remove date
full_x = train_x.copy()[:-1]
full_x_oh = train_x_oh.copy()[:-1]
full_y = target_list[1][:-1]
del train_x, train_x_oh




# # train test split
# train 2021/1/6 ~ 2021/9/5
# validation 2021/9/6 ~ 2021/9/17
# test 2021/9/27 ~ 2021/10/1

train_x = full_x[full_x["date"] < datetime(2021, 9, 6)]
train_x_oh = full_x_oh[full_x["date"] < datetime(2021, 9, 6)]
train_y = full_y[full_x["date"] < datetime(2021, 9, 6)]

val_x = full_x[(full_x["date"] >= datetime(2021, 9, 6)) & (full_x["date"] < datetime(2021, 9, 18))]
val_x_oh = full_x_oh[(full_x["date"] >= datetime(2021, 9, 6)) & (full_x["date"] < datetime(2021, 9, 18))]
val_y = full_y[(full_x["date"] >= datetime(2021, 9, 6)) & (full_x["date"] < datetime(2021, 9, 18))]

test_x = full_x[full_x["date"] >= datetime(2021, 9, 27)]
test_x_oh = full_x_oh[full_x["date"] >= datetime(2021, 9, 27)]
test_y = full_y[full_x["date"] >= datetime(2021, 9, 27)]

full_x.shape[0] == train_x.shape[0] + val_x.shape[0] + test_x.shape[0] + 5
full_y.shape[0] == train_y.shape[0] + val_y.shape[0] + test_y.shape[0] + 5

full_x.drop("date", axis=1, inplace=True)
full_x_oh.drop("date", axis=1, inplace=True)
train_x.drop("date", axis=1, inplace=True)
train_x_oh.drop("date", axis=1, inplace=True)
val_x.drop("date", axis=1, inplace=True)
val_x_oh.drop("date", axis=1, inplace=True)
test_x.drop("date", axis=1, inplace=True)
test_x_oh.drop("date", axis=1, inplace=True)





# from sklearn.model_selection import TimeSeriesSplit
#
# cv_spliter = TimeSeriesSplit(n_splits=2, max_train_size=20, test_size=1, gap=0)
#
# full_x.shape
# for train_idx, val_idx in cv_spliter.split(full_x):
#     print("TRAIN:", train_idx, "TEST:", val_idx)
#     print(len(train_idx))
#     # X_train, X_test = X[train_index], X[val_idx]
#     # y_train, y_test = y[train_index], y[val_idx]


# ds = {"full_x": full_x, "full_x_oh": full_x_oh, "full_y": full_y,
#       "train_x": train_x, "train_x_oh": train_x_oh, "train_y": train_y,
#       "val_x": val_x, "val_x_oh": val_x_oh, "val_y": val_y,
#       "test_x": test_x, "test_x_oh": test_x_oh,
#       "cat_vars": cat_vars, "onehot_encoder": onehot_encoder}

model_names = ["Linear", "ElasticNet", "SVM", "XGB_GBT", "LGB_RF", "LGB_GOSS", "CAT_GBM", "KNN",
               "MLP_Desc_V1", "MLP_ResNet_V1", "MLP_DenseNet_V1", "MLP_LP_V1", "MLP_MultiActs_V1",
               "StackingEnsemble"]

result_val = dict.fromkeys(model_names)
for i in result_val.keys(): result_val[i] = {}
result_test = dict.fromkeys(model_names)
for i in result_test.keys(): result_test[i] = {}

# # === save obejcts ===
# easyIO(ds, folder_path + "dataset/ds.pickle", op="w")
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

# load obejcts
result_val = easyIO(None, folder_path + "dataset/result_val.pickle", op="r")
result_test = easyIO(None, folder_path + "dataset/result_test.pickle", op="r")
ds = easyIO(None, folder_path + "dataset/ds.pickle", op="r")

kfolds_spliter = StratifiedKFold(10, shuffle=True, random_state=8899)

targetType = "numeric"
targetTask = None
cut_off = 0
# cut_off = round(ds["train_y"].mean(), 3)


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
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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

result_val["Linear"] = doLinear(ds["train_x_oh"], ds["train_y"],
                                ds["val_x_oh"], ds["val_y"],
                                model_export=True)
print(result_val["Linear"]["performance"])

result_test["Linear"] = doLinear(ds["train_x_oh"], ds["train_y"],
                                 ds["test_x_oh"], None,
                                 preTrained=result_val["Linear"]["model"])
print(result_val["Linear"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

def doElasticNet(train_x, train_y, test_x=None, test_y=None, kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=1000):
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
            model_tuner = GridTuner(lm.ElasticNet(max_iter=1000, normalize=False, random_state=seed),
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
                                                normalize=False, random_state=seed)

        result_dic["model"].fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_standard.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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
                                                            max_iter=1000,
                                                            C=model_tuner.best_params_["C"],
                                                            l1_ratio=model_tuner.best_params_["l1_ratio"])

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

result_val["ElasticNet"] = doElasticNet(ds["train_x_oh"], ds["train_y"],
                                        ds["val_x_oh"], ds["val_y"],
                                        kfolds=kfolds_spliter,
                                        model_export=True)
print(result_val["ElasticNet"]["performance"])
print(result_val["ElasticNet"]["running_time"])

result_test["ElasticNet"] = doElasticNet(ds["full_x_oh"], ds["full_y"],
                                         ds["test_x_oh"], None,
                                         preTrained=result_val["ElasticNet"]["model"])
print(result_val["ElasticNet"]["pred"][:10])

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

result_val["QDA"] = doQDA(ds["train_x_oh"], ds["train_y"],
                          ds["val_x_oh"], ds["val_y"],
                          model_export=True)
print(result_val["QDA"]["performance"])
print(result_val["QDA"]["running_time"])

result_test["QDA"] = doQDA(ds["full_x_oh"], ds["full_y"],
                           ds["test_x_oh"], None,
                           preTrained=result_val["QDA"]["model"])
print(result_val["QDA"]["pred"][:10])



def doSVM(train_x, train_y, test_x=None, test_y=None,
          kernelSeq=["linear", "poly", "rbf"],
          costSeq=[pow(10, i) for i in [-2, -1, 0, 1, 2]],
          gammaSeq=[pow(10, i) for i in [-2, -1, 0, 1, 2]],
          kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=7777):
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
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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

result_val["SVM"] = doSVM(ds["train_x_oh"], ds["train_y"],
                          ds["val_x_oh"], ds["val_y"],
                          kfolds=kfolds_spliter,
                          model_export=True)
print(result_val["SVM"]["best_params"])
print(result_val["SVM"]["performance"])
print(result_val["SVM"]["running_time"])

result_test["SVM"] = doSVM(ds["full_x_oh"], ds["full_y"],
                           ds["test_x_oh"], None,
                           preTrained=result_val["SVM"]["model"])
print(result_test["SVM"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


# display the performance
for k, v in result_val.items():
    if v is not None:
        if "performance" in v.keys():
            print(k, "--->", v["performance"])



def doXGB(train_x, train_y, test_x=None, test_y=None, ntrees=5000, eta=5e-3,
          depthSeq=[4, 6, 8], subsampleSeq=[0.6, 0.8], colsampleSeq=[0.6, 0.8],
          l2Seq=[0.1, 1.0, 5.0], mcwSeq=[1, 3, 5], gammaSeq=[0.0, 0.2],
          kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=11):
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

        result_dic["model"].fit(train_x, train_y, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
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


# xgboost - gbtree
result_val["XGB_GBT"] = doXGB(ds["train_x_oh"], ds["train_y"],
                              ds["val_x_oh"], ds["val_y"],
                              kfolds=kfolds_spliter,
                              ntrees=5000,
                              model_export=True)
print(result_val["XGB_GBT"]["best_params"])
print(result_val["XGB_GBT"]["performance"])
print(result_val["XGB_GBT"]["running_time"])

# xgboost - gbtree
result_test["XGB_GBT"] = doXGB(ds["full_x_oh"], ds["full_y"],
                               ds["test_x_oh"], None,
                               preTrained=result_val["XGB_GBT"]["model"])
print(result_test["XGB_GBT"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


def doLGB(train_x, train_y, test_x=None, test_y=None, categoIdx=None, boostingType="goss", ntrees=5000, eta=5e-3,
          leavesSeq=[pow(2, i) - 1 for i in [4, 6, 8]], subsampleSeq=[0.6, 0.8], gammaSeq=[0.0, 0.2],
          colsampleSeq=[0.6, 0.8], l2Seq=[0.1, 1.0, 5.0], mcsSeq=[5, 10, 20], mcwSeq=[1e-4, 1e-3, 1e-2],
          kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=22):
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

        result_dic["model"].fit(train_x, train_y, categorical_feature=categoIdx, verbose=False)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
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

# lightgbm - randomforest
result_val["LGB_RF"] = doLGB(ds["train_x"], ds["train_y"],
                             ds["val_x"], ds["val_y"],
                             categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                             kfolds=kfolds_spliter,
                             boostingType="rf", ntrees=500,
                             model_export=True)
print(result_val["LGB_RF"]["best_params"])
print(result_val["LGB_RF"]["performance"])
print(result_val["LGB_RF"]["running_time"])

result_test["LGB_RF"] = doLGB(ds["full_x"], ds["full_y"],
                              ds["test_x"], None,
                              categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                              preTrained=result_val["LGB_RF"]["model"])
print(result_test["LGB_RF"]["pred"][:10])

# lightgbm - goss
result_val["LGB_GOSS"] = doLGB(ds["train_x"], ds["train_y"],
                               ds["val_x"], ds["val_y"],
                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                               kfolds=kfolds_spliter,
                               boostingType="goss", ntrees=5000,
                               model_export=True)
print(result_val["LGB_GOSS"]["best_params"])
print(result_val["LGB_GOSS"]["performance"])
print(result_val["LGB_GOSS"]["running_time"])


result_test["LGB_GOSS"] = doLGB(ds["full_x"], ds["full_y"],
                                ds["test_x"], None,
                                categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                                preTrained=result_val["LGB_GOSS"]["model"])
print(result_test["LGB_GOSS"]["pred"][:10])


# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


def doCAT(train_x, train_y, test_x=None, test_y=None, categoIdx=None, boostingType="Plain", ntrees=5000, eta=5e-2,
          depthSeq=[4, 6, 8], bagTempSeq=[0.2, 0.8], colsampleSeq=[0.6, 0.8], l2Seq=[0.1, 1.0, 5.0], random_strength=[0.1, 1.0],
          kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=33):
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

        result_dic["model"].fit(train_x, train_y, cat_features=categoIdx)
        if test_x is not None:
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
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

                result_dic["model"] = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="Logloss",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                            rsm=model_tuner.best_params_["rsm"],
                                                            l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                            random_strength=model_tuner.best_params_["random_strength"],
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

                result_dic["model"] = cat.CatBoostRegressor(boosting_type=boostingType, loss_function="MultiClass",
                                                            n_estimators=result_dic["best_params"]["best_trees"], learning_rate=eta,
                                                            max_depth=model_tuner.best_params_["max_depth"],
                                                            bagging_temperature=model_tuner.best_params_["bagging_temperature"],
                                                            rsm=model_tuner.best_params_["rsm"],
                                                            l2_leaf_reg=model_tuner.best_params_["l2_leaf_reg"],
                                                            random_strength=model_tuner.best_params_["random_strength"],
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

# CatBoost - GBM
result_val["CAT_GBM"] = doCAT(ds["train_x"], ds["train_y"],
                              ds["val_x"], ds["val_y"],
                              categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                              kfolds=kfolds_spliter,
                              boostingType="Plain", ntrees=5000,
                              model_export=True)

print(result_val["CAT_GBM"]["best_params"])
print(result_val["CAT_GBM"]["performance"])
print(result_val["CAT_GBM"]["running_time"])

result_test["CAT_GBM"] = doCAT(ds["full_x"], ds["full_y"],
                               ds["test_x"], None,
                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                               preTrained=result_val["CAT_GBM"]["model"])
print(result_test["CAT_GBM"]["pred"][:10])



# CatBoost - Ordered Boosting
result_val["CAT_ORD"] = doCAT(ds["train_x"], ds["train_y"],
                              ds["val_x"], ds["val_y"],
                              categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                              boostingType="Ordered", ntrees=5000,
                              model_export=True)

print(result_val["CAT_ORD"]["best_params"])
print(result_val["CAT_ORD"]["performance"])
print(result_val["CAT_ORD"]["running_time"])

result_test["CAT_ORD"] = doCAT(ds["full_x"], ds["full_y"],
                               ds["test_x"], None,
                               categoIdx=findIdx(ds["full_x"], ds["cat_vars"]),
                               preTrained=result_val["CAT_ORD"]["model"])
print(result_test["CAT_ORD"]["pred"][:10])


# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


for k, v in result_val.items():
    if v is not None:
        if "cv_score" in v.keys():
            print("===", k, "===\n", v["cv_score"])


def doKNN(train_x, train_y, test_x=None, test_y=None, kSeq=[3, 5, 7], kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=7777):
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

        result_dic["model"] = knn_model.fit(train_x, train_y)
        if test_x is not None:
            test_x = scaler_minmax.transform(test_x)
            result_dic["pred"] = result_dic["model"].predict(test_x)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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

            result_dic["model"] = knn_model.fit(train_x, train_y)
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

            result_dic["model"] = knn_model.fit(train_x, train_y)
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


kSeq = list(range(3, int(ds["train_x"].shape[0] * 0.05), 2))

result_val["KNN"] = doKNN(ds["train_x_oh"], ds["train_y"],
                          ds["val_x_oh"], ds["val_y"],
                          kfolds=kfolds_spliter,
                          kSeq=kSeq, model_export=True)
print(result_val["KNN"]["best_params"])
print(result_val["KNN"]["performance"])
print(result_val["KNN"]["running_time"])

result_test["KNN"] = doKNN(ds["full_x"], ds["full_y"],
                           ds["test_x"], None,
                           preTrained=result_val["KNN"]["model"])
print(result_test["KNN"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

for k, v in result_val.items():
    if v is not None:
        if "performance" in v.keys():
            print(k, "--->", v["performance"])



class MyHyperModel(HyperModel):
    def __init__(self, nCols, mlpName, hiddenLayers, dropoutRate, seqLength=0, eta=1e-3):
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
def createNetwork(nCols, mlpName, hiddenLayers=128, dropoutRate=0.2, seqLength=5, eta=3e-3):
    # nCols = 10
    # hiddenLayers = 128
    # dropoutRate = 0.2
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
def doMLP(train_x, train_y, test_x, test_y, mlpName="MLP_Desc_V1",
          hiddenLayers={"min": 32, "max": 128, "step": 32}, dropoutRate=0.2,
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
            result_dic["pred"] = result_dic["model"].predict(test_x, batch_size=batch_size)
            if test_y is not None:
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
                                             "R2": metrics.r2_score(test_y, result_dic["pred"])}
            else:
                result_dic["performance"] = None
    else:
        if targetTask == "binary":
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
        else:
            if preTrained is not None:
                result_dic["model"] = preTrained
            else:
                encoder_onehot = OneHotEncoder(sparse=False)
                train_y_sparse = encoder_onehot.fit_transform(train_y[..., np.newaxis])
                test_y_sparse = encoder_onehot.transform(test_y[..., np.newaxis])

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

input_epochs = 200
input_batch_size = 4

input_hiddenLayers = {"min": 64, "max": 512, "step": 64}

# === MLP_Desc_V1 ===
result_val["MLP_Desc_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
                                  ds["val_x_oh"], ds["val_y"],
                                  mlpName="MLP_Desc_V1", epochs=input_epochs, batch_size=input_batch_size,
                                  hiddenLayers=input_hiddenLayers, model_export=True)
print(result_val["MLP_Desc_V1"]["model"].summary())
print(result_val["MLP_Desc_V1"]["best_params"])
print(result_val["MLP_Desc_V1"]["performance"])
print(result_val["MLP_Desc_V1"]["running_time"])

result_test["MLP_Desc_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
                                   ds["test_x_oh"], None,
                                   preTrained=result_val["MLP_Desc_V1"]["model"])
print(result_val["MLP_Desc_V1"]["pred"][:10])

# === MLP_ResNet_V1 ===
result_val["MLP_ResNet_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
                                    ds["val_x_oh"], ds["val_y"],
                                    mlpName="MLP_ResNet_V1", epochs=input_epochs, batch_size=input_batch_size,
                                    hiddenLayers=input_hiddenLayers, model_export=True)
print(result_val["MLP_ResNet_V1"]["model"].summary())
print(result_val["MLP_ResNet_V1"]["best_params"])
print(result_val["MLP_ResNet_V1"]["performance"])
print(result_val["MLP_ResNet_V1"]["running_time"])

result_test["MLP_ResNet_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
                                     ds["test_x_oh"], None,
                                     preTrained=result_val["MLP_ResNet_V1"]["model"])
print(result_val["MLP_ResNet_V1"]["pred"][:10])

# === MLP_DenseNet_V1 ===
result_val["MLP_DenseNet_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
                                      ds["val_x_oh"], ds["val_y"],
                                      mlpName="MLP_DenseNet_V1", epochs=input_epochs, batch_size=input_batch_size,
                                      hiddenLayers=input_hiddenLayers, model_export=True)
print(result_val["MLP_DenseNet_V1"]["model"].summary())
print(result_val["MLP_DenseNet_V1"]["best_params"])
print(result_val["MLP_DenseNet_V1"]["performance"])
print(result_val["MLP_DenseNet_V1"]["running_time"])

result_test["MLP_DenseNet_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
                                       ds["test_x_oh"], None,
                                       preTrained=result_val["MLP_DenseNet_V1"]["model"])
print(result_val["MLP_DenseNet_V1"]["pred"][:10])

# === MLP_LP_V1 ===
result_val["MLP_LP_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
                                ds["val_x_oh"], ds["val_y"],
                                mlpName="MLP_LP_V1", epochs=input_epochs, batch_size=input_batch_size,
                                hiddenLayers=input_hiddenLayers, model_export=True)
print(result_val["MLP_LP_V1"]["model"].summary())
print(result_val["MLP_LP_V1"]["best_params"])
print(result_val["MLP_LP_V1"]["performance"])
print(result_val["MLP_LP_V1"]["running_time"])

result_test["MLP_LP_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
                                 ds["test_x_oh"], None,
                                 preTrained=result_val["MLP_LP_V1"]["model"])
print(result_val["MLP_LP_V1"]["pred"][:10])

# === MLP_MultiActs_V1 ===
result_val["MLP_MultiActs_V1"] = doMLP(ds["train_x_oh"], ds["train_y"],
                                       ds["val_x_oh"], ds["val_y"],
                                       mlpName="MLP_MultiActs_V1", epochs=input_epochs, batch_size=input_batch_size,
                                       hiddenLayers=input_hiddenLayers, model_export=True)
print(result_val["MLP_MultiActs_V1"]["model"].summary())
print(result_val["MLP_MultiActs_V1"]["best_params"])
print(result_val["MLP_MultiActs_V1"]["performance"])
print(result_val["MLP_MultiActs_V1"]["running_time"])

result_test["MLP_MultiActs_V1"] = doMLP(ds["full_x_oh"], ds["full_y"],
                                        ds["test_x_oh"], None,
                                        preTrained=result_val["MLP_MultiActs_V1"]["model"])
print(result_val["MLP_MultiActs_V1"]["pred"][:10])



# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

# display the performance



# Stacking Ensemble
seed_base_models = 933

# if regression
# result_val["MLP_Desc_V1"]["model"].trainable = False
# result_val["MLP_ResNet_V1"]["model"].trainable = False
# result_val["MLP_MultiActs_V1"]["model"].trainable = False

# CAT_GBM   0.355392  0.837989  0.805369  0.920356
# XGB_GBT   0.355467  0.854749  0.824324  0.917260
# LGB_RF
#
# MLP_ResNet_V1        NaN  0.871508  0.836879  0.913570
# MLP_LP_V1   0.368466  0.849162  0.805755  0.913307
# MLP_MultiActs_V1   0.367828  0.871508  0.834532  0.912385
# LGB_GOSS   0.377965  0.832402  0.794521  0.911331

# result_val["MLP_ResNet_V1"]["model"].trainable = False
# result_val["MLP_LP_V1"]["model"].trainable = False
# result_val["MLP_MultiActs_V1"]["model"].trainable = False
stacking_base_models = [
    # ('ElasticNet', make_pipeline(StandardScaler(), lm.ElasticNet(alpha=result_val["ElasticNet"]["best_params"]["alpha"],
    #                                                              l1_ratio=result_val["ElasticNet"]["best_params"]["l1_ratio"],
    #                                                              normalize=False, random_state=seed_base_models+1))),
    # ('SVM', make_pipeline(MinMaxScaler(), svm.SVR(kernel=result_val["SVM"]["best_params"]["kernel"],
    #                                               C=result_val["SVM"]["best_params"]["C"],
    #                                               gamma=result_val["SVM"]["best_params"]["gamma"]))),
    ('XGB_GBT', xgb.XGBClassifier(booster="gbtree", objective="binary:logistic", learning_rate=5e-3,
                                  n_estimators=result_val["XGB_GBT"]["model"].best_iteration,
                                  max_depth=result_val["XGB_GBT"]["best_params"]["max_depth"],
                                  subsample=result_val["XGB_GBT"]["best_params"]["subsample"],
                                  reg_lambda=result_val["XGB_GBT"]["best_params"]["reg_lambda"],
                                  min_child_weight=result_val["XGB_GBT"]["best_params"]["min_child_weight"],
                                  gamma=result_val["XGB_GBT"]["best_params"]["gamma"],
                                  colsample_bytree=0.8, verbosity=0, use_label_encoder=False,
                                  n_jobs=None, random_state=seed_base_models+2)),
    ('LGB_RF', lgb.LGBMClassifier(boosting_type="rf", objective="binary", learning_rate=5e-3,
                                  n_estimators=result_val["LGB_RF"]["model"].best_iteration_,
                                  num_leaves=result_val["LGB_RF"]["best_params"]["num_leaves"],
                                  subsample=result_val["LGB_RF"]["best_params"]["subsample"],
                                  reg_lambda=result_val["LGB_RF"]["best_params"]["reg_lambda"],
                                  min_child_weight=result_val["LGB_RF"]["best_params"]["min_child_weight"],
                                  min_child_samples=result_val["LGB_RF"]["best_params"]["min_child_samples"],
                                  min_split_gain=result_val["LGB_RF"]["best_params"]["min_split_gain"],
                                  subsample_freq=2, colsample_bytree=0.8, silent=True,
                                  n_jobs=None, random_state=seed_base_models+3)),
    # ('LGB_GOSS', lgb.LGBMClassifier(boosting_type="goss", objective="binary", learning_rate=5e-3,
    #                                 n_estimators=result_val["LGB_GOSS"]["model"].best_iteration_,
    #                                 num_leaves=result_val["LGB_GOSS"]["best_params"]["num_leaves"],
    #                                 subsample=result_val["LGB_GOSS"]["best_params"]["subsample"],
    #                                 reg_lambda=result_val["LGB_GOSS"]["best_params"]["reg_lambda"],
    #                                 min_child_weight=result_val["LGB_GOSS"]["best_params"]["min_child_weight"],
    #                                 min_child_samples=result_val["LGB_GOSS"]["best_params"]["min_child_samples"],
    #                                 min_split_gain=result_val["LGB_GOSS"]["best_params"]["min_split_gain"],
    #                                 colsample_bytree=0.8, silent=True,
    #                                 n_jobs=None, random_state=seed_base_models+4)),
    ('CAT_GBM', cat.CatBoostClassifier(boosting_type="Plain", loss_function="Logloss", learning_rate=5e-2,
                                       n_estimators=result_val["CAT_GBM"]["model"].best_iteration_,
                                       max_depth=result_val["CAT_GBM"]["best_params"]["max_depth"],
                                       bagging_temperature=result_val["CAT_GBM"]["best_params"]["bagging_temperature"],
                                       l2_leaf_reg=result_val["CAT_GBM"]["best_params"]["l2_leaf_reg"],
                                       random_strength=result_val["CAT_GBM"]["best_params"]["random_strength"],
                                       rsm=0.8, logging_level="Silent",
                                       thread_count=None, random_state=seed_base_models+5)),
    # ("MLP_ResNet_V1", KerasClassifier(model=result_val["MLP_ResNet_V1"]["model"], batch_size=4, shuffle=False,
    #                                   verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50)),
    # ("MLP_LP_V1", KerasClassifier(model=result_val["MLP_LP_V1"]["model"], batch_size=4, shuffle=False,
    #                               verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50)),
    # ("MLP_MultiActs_V1", KerasClassifier(model=result_val["MLP_MultiActs_V1"]["model"], batch_size=4, shuffle=False,
    #                                      verbose=0, fit__use_multiprocessing=False, random_state=seed_base_models+50))
]


# meta learner definition
# meta_learner_model = lm.LinearRegression()
# meta_learner_model = lm.ElasticNetCV(cv=10, alphas=np.linspace(1e-3, 1e+3, 100).tolist(), random_state=seed_base_models+100,
#                                      l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])
meta_learner_model = lm.LogisticRegressionCV(penalty="elasticnet", solver="saga", multi_class="ovr", cv=10, max_iter=1000, random_state=4,
                                             Cs=np.linspace(1e-3, 1e+3, 100).tolist(),
                                             l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])


def doStackingEnsemble(train_x, train_y, test_x, test_y, stacking_base_models, meta_learner_model,
                       kfolds=KFold(10, True, 2323), model_export=False, preTrained=None, seed=7788):
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
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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


result_val["StackingEnsemble"] = doStackingEnsemble(ds["train_x_oh"], ds["train_y"],
                                                    ds["val_x_oh"], ds["val_y"],
                                                    stacking_base_models=stacking_base_models,
                                                    meta_learner_model=meta_learner_model,
                                                    kfolds=kfolds_spliter, model_export=True)
print(result_val["StackingEnsemble"]["performance"])


result_test["StackingEnsemble"] = doStackingEnsemble(ds["train_x_oh"], ds["train_y"],
                                                     ds["test_x_oh"], None,
                                                     stacking_base_models=None, meta_learner_model=None,
                                                     preTrained=result_val["StackingEnsemble"]["model"])
print(result_test["StackingEnsemble"]["pred"][:10])

# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")

# display the performance
for k, v in result_val.items():
    if v is not None:
        if "performance" in v.keys():
            print(k, "--->", v["performance"])


# K-folds MLP ensemble
class kfoldsMLP:
    def __init__(self, nCols, mlpName, hiddenLayers, dropoutRate, seqLength=5, boosters={}, meta_learner=None, ntrees=5000, earlyStopping=100):
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
                    self.performanceWB["Total"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, self.predWB)),
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
                    self.performance = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, self.pred)),
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
                result_dic["performance"] = {"RMSE": np.sqrt(metrics.mean_squared_error(test_y, result_dic["pred"])),
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

input_epochs = 200
input_batch_size = 4

mlpWB_boosters = {"XGB_GBT": {"booster": None},
                  "LGB_GOSS": {"booster": None},
                  "CAT_GBM": {"booster": None}}
# mlpWB_boosters = {"XGB_GBT": {"booster": xgb.XGBClassifier(booster="gbtree", objective="binary:logistic", colsample_bytree=0.8,
#                                                            n_estimators=mlpWB_ntrees, random_state=1,
#                                                            max_depth=result_val["XGB_GBT"]["best_params"]["max_depth"],
#                                                            subsample=result_val["XGB_GBT"]["best_params"]["subsample"],
#                                                            reg_lambda=result_val["XGB_GBT"]["best_params"]["reg_lambda"],
#                                                            min_child_weight=result_val["XGB_GBT"]["best_params"]["min_child_weight"],
#                                                            gamma=result_val["XGB_GBT"]["best_params"]["gamma"],
#                                                            learning_rate=5e-3, n_jobs=multiprocessing.cpu_count(), verbosity=0)},
#                   "LGB_GOSS": {"booster": lgb.LGBMClassifier(boosting_type="goss", objective="binary", colsample_bytree=0.8,
#                                                              n_estimators=mlpWB_ntrees, random_state=2,
#                                                              num_leaves=result_val["LGB_GOSS"]["best_params"]["num_leaves"],
#                                                              subsample=result_val["LGB_GOSS"]["best_params"]["subsample"],
#                                                              reg_lambda=result_val["LGB_GOSS"]["best_params"]["reg_lambda"],
#                                                              min_child_weight=result_val["LGB_GOSS"]["best_params"]["min_child_weight"],
#                                                              min_child_samples=result_val["LGB_GOSS"]["best_params"]["min_child_samples"],
#                                                              min_split_gain=result_val["LGB_GOSS"]["best_params"]["min_split_gain"],
#                                                              learning_rate=5e-3, n_jobs=multiprocessing.cpu_count(), silent=True)},
#                   "CAT_GBM": {"booster": cat.CatBoostClassifier(boosting_type="Plain", loss_function="Logloss", rsm=0.8,
#                                                                 n_estimators=mlpWB_ntrees, random_state=3,
#                                                                 max_depth=result_val["CAT_GBM"]["best_params"]["max_depth"],
#                                                                 bagging_temperature=result_val["CAT_GBM"]["best_params"]["bagging_temperature"],
#                                                                 l2_leaf_reg=result_val["CAT_GBM"]["best_params"]["l2_leaf_reg"],
#                                                                 random_strength=result_val["CAT_GBM"]["best_params"]["random_strength"],
#                                                                 learning_rate=5e-2, thread_count=multiprocessing.cpu_count(), logging_level="Silent")}}

# mlpWB_metaLearner = lm.ElasticNetCV(cv=10, max_iter=1000, alphas=np.linspace(1e-3, 1e+3, 100).tolist(), random_state=4,
#                                     l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])
mlpWB_metaLearner = lm.LogisticRegressionCV(penalty="elasticnet", solver="saga", multi_class="ovr", cv=kfolds_spliter, max_iter=1000, random_state=4,
                                            Cs=np.linspace(1e-3, 1e+3, 100).tolist(),
                                            l1_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.99])

# kfoldsMLP_ResNet_V1
result_val["kfoldsMLP_ResNet_V1"] = {}
result_test["kfoldsMLP_ResNet_V1"] = {}

input_hiddenLayers = 64
# input_hiddenLayers = result_val["MLP_ResNet_V1"]["best_params"]["hiddenLayers"]

result_val["kfoldsMLP_ResNet_V1"]["model"] = kfoldsMLP(nCols=ds["full_x_oh"].shape[1],
                                                       boosters=mlpWB_boosters, meta_learner=mlpWB_metaLearner,
                                                       ntrees=5000, earlyStopping=int(5000*0.2), mlpName="MLP_ResNet_V1",
                                                       hiddenLayers=input_hiddenLayers, dropoutRate=0.2)

result_val["kfoldsMLP_ResNet_V1"]["model"].fit_predict(ds["full_x_oh"], ds["full_y"],
                                                       ds["test_x_oh"], None,
                                                       epochs=input_epochs, batch_size=input_batch_size,
                                                       kfolds=5, stratify=ds["full_y"],
                                                       seed=7373, withBoosting=True, model_export=True)

result_val["kfoldsMLP_ResNet_V1"]["performance"] = result_val["kfoldsMLP_ResNet_V1"]["model"].performanceWB["Total"]
print(result_val["kfoldsMLP_ResNet_V1"]["performance"])

result_test["kfoldsMLP_ResNet_V1"] = result_val["kfoldsMLP_ResNet_V1"]["model"].predict(ds["test_x_oh"], None, batch_size=input_batch_size)
print(result_test["kfoldsMLP_ResNet_V1"]["pred"][:10])



# # save obejcts
# easyIO(result_val, folder_path + "dataset/result_val.pickle", op="w")
# easyIO(result_test, folder_path + "dataset/result_test.pickle", op="w")


# display the performance
for k, v in result_val.items():
    if v is not None:
        if "performance" in v.keys():
            print(k, "--->", v["performance"])

# ===== Submission =====

sub_select = ["LGB_RF", "XGB_GBT", "CAT_GBM", "StackingEnsemble"]

submission = read_csv("./kaggle_housingprice/house-prices-advanced-regression-techniques/sample_submission.csv")

for i in sub_select:
    submission["SalePrice"] = [int(round(i, 0)) for i in np.expm1(result_test[i]["pred"])]
    submission.to_csv("./kaggle_housingprice/submission/sub2_" + i + ".csv", index=False)
