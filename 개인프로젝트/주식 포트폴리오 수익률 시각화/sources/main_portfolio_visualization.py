import os
import sys
sys.path.append("projects/DA_Platform")
from DA_v5 import *
from multiprocessing import cpu_count
import copy
import pickle
import warnings
from datetime import datetime, timedelta
from time import time, sleep, mktime, tzname
from matplotlib import font_manager as fm, rc, rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
import random

import numpy as np
from numpy import array, nan, random as rnd, where
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset
import statsmodels.api as sm
from scipy.stats import f_oneway

from sklearn import preprocessing as prep
from sklearn.preprocessing import KBinsDiscretizer
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
from optuna import distributions as optuna_dist, visualization as optuna_plt, Trial, create_study
from optuna.integration import OptunaSearchCV
from optuna.samplers import TPESampler
from optuna.logging import set_verbosity as optuna_set_verbose

# ===== tensorflow =====
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import metrics as tf_metrics
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa
import keras_tuner as kt
from keras_tuner import HyperModel
from tensorflow.keras.utils import plot_model

# ===== timeseries =====
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.preprocessing import timeseries_dataset_from_array as make_ts_tensor

# # ===== NLP =====
# from selenium import webdriver
# from konlpy.tag import Okt
# from KnuSentiLex.knusl import KnuSL

# ===== utility functions =====
# label encoding for categorical column with excepting na value
# # 괄호를 포함한 괄호 안 문자 제거 정규식
# re.sub(r'\([^)]*\)', '', remove_text)
# # <>를 포함한 <> 안 문자 제거 정규식
# re.sub(r'\<[^)]*\>', '', remove_text)

# global setting
warnings.filterwarnings(action='ignore')
rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

font_path = 'myfonts/NanumSquareB.ttf'
font_obj = fm.FontProperties(fname=font_path, size=12).get_name()
rc('font', family=font_obj)




# folder_path = "projects/dacon_stockprediction/"
seed_everything()

from scipy import stats
# yahoo finance는 기본적으로 영업일을 모두 가져옴
# 영업일이지만 휴장일이면 nan값을 가짐
import yfinance as yf
# time zone information
import pytz
# list comprehension for selecting 'Asia****' time zones
tz_list = [x for x in pytz.common_timezones if re.compile(r'^US').match(x)]
timezone = 'US/Eastern'

# ===== preprocessing & feature engineering =====

sector_names = ["IT", "Semiconductor", "Communication Service", "Discretionary", "Staples",
                "Health Care", "Financials", "Industrials", "Alternatives"]
ref_code_change = {"RDS/B": "RDS-B",
                   "SNE": "SONY",
                   "005935": "005935.KS", "5935": "005935.KS"}

# 거래 내역 일자 범위 : 2020.10.01 ~ 2022.01.28
trading_history = read_csv("projects/portfolio_visualization/통합_거래내역_20201001_20220128.csv", parse_dates=["매매일자"])
trading_history.head(10)
trading_history.tail(10)
trading_history.info()
trading_history.sort_values(by=["매매일자"], inplace=True)
trading_history.set_index("매매일자", inplace=True)
trading_history.index.name = "date"
trading_history["매매구분"].replace({"현금매수": "매수", "현금매도": "매도", "매도정정*": "매도"}, inplace=True)
trading_history["종목코드"].replace(ref_code_change, inplace=True)

# update stock information table
table_stock_info = read_csv("projects/portfolio_visualization/stock_info.csv", encoding="euc-kr")
tmp_stock_info = dataframe(columns=["code", "name", "sector"])

tmp_stock_info["code"] = trading_history["종목코드"].drop_duplicates().reset_index(drop=True)
tmp_stock_info["name"] = [yf.Ticker(i).info["longName"] if i not in table_stock_info["code"].values else nan for i in tmp_stock_info["code"].values]
tmp_stock_info = tmp_stock_info[~(tmp_stock_info["name"].isna())]
tmp_stock_info["sector"] = "Unknown"
table_stock_info = pd.concat([table_stock_info, tmp_stock_info], axis=0, ignore_index=True)
table_stock_info.to_csv("projects/portfolio_visualization/stock_info.csv", index=False)
table_stock_info.to_csv("projects/portfolio_visualization/stock_info_" + datetime.now().strftime("%Y%m%d") +".csv", index=False)

trading_history.drop("종목명", axis=1, inplace=True)

# 초기 포트폴리오 일자 : 2020.9.30
pf_init = read_csv("projects/portfolio_visualization/체결잔고_20200930.csv", encoding="euc-kr")
pf_init.head(10)
pf_init["종목번호"].replace(ref_code_change, inplace=True)

# 환율 dataframe 생성
rawdata_krwtousd = yf.download("KRW=X", start="2020-09-30", end="2022-01-28")
krwtousd = rawdata_krwtousd["Close"]
krwtousd.head(10)

trading_history.columns = ["type", "code", "shares", "price", "currency"]
pf_init.columns = ["code", "shares", "price_avg"]

# transform all prices from krw to usd
trading_history["price"] = [round(trading_history["price"][idx] / krwtousd[trading_history.index[idx]], 2) if value == "KRW" else trading_history["price"][idx] for idx, value in enumerate(trading_history["currency"])]
pf_init["price_avg"] = round(pf_init["price_avg"] / krwtousd["2020-09-30"], 2)
table_currency = trading_history[["code", "currency"]].drop_duplicates().reset_index(drop=True)
pf_init["total_cost"] = pf_init["shares"] * pf_init["price_avg"]

# loading the stock price
table_stockprice = yf.download(" ".join(set(trading_history["code"].to_list() + pf_init["code"].to_list() + ["^GSPC"])),
                               start="2020-09-30", end="2022-01-29")["Close"]
table_stockprice.index.name = "date"
table_stockprice.head(10)
table_stockprice.tail(10)
# drop holidays
table_stockprice = table_stockprice[~(table_stockprice["^GSPC"].isna())]
table_stockprice.ffill(inplace=True)
table_stockprice.bfill(inplace=True)
table_stockreturn = table_stockprice.pct_change()

# converting currency KRW to USD
tmp = []
for i, j in zip(table_currency["code"], table_currency["currency"]):
    if j == "KRW":
        for m, n in enumerate(table_stockprice[i]):
            tmp.append(round(n / krwtousd[krwtousd.index[m]], 2))
        table_stockprice[i] = tmp
trading_history.drop("currency", axis=1, inplace=True)
trading_history.head(10)


init_date = datetime(2020, 10, 1)
pf_daily = pf_init.copy()
table_closed = dataframe(columns=["date", "code", "price_cost", "price_sell", "shares_sell", "profit", "return"])
table_compare = dataframe(index=table_stockprice.index[table_stockprice.index >= init_date])
# table_sector = dataframe(columns=["value_" + i for i in sector_names] + ["weight_" + i for i in sector_names] + ["return_" + i for i in sector_names])
table_sector = dataframe()
tmp_pf_marketvalue = [sum(series([table_stockprice[corp_code][init_date - timedelta(days=1)] for corp_code in pf_init["code"]]).values * pf_init["shares"].values)]
tmp_price_weighted_return = []

for i in pd.date_range(start=table_stockprice.index[0], end=table_stockprice.index[-1]):
    if i < init_date:
        continue

    # make holdings table
    tmp_history = trading_history[trading_history.index == i-timedelta(days=1)]
    print("portfolio update date (from -1 day)", i)

    # if trading exists
    if tmp_history.shape[0] != 0:
        # update long position
        tmp_position = tmp_history[tmp_history["type"] == "매수"]
        for idx, value in enumerate(tmp_position["code"]):
            # already exits on holdings
            if value in pf_daily["code"].values:
                pf_daily["shares"][pf_daily["code"] == value] += tmp_position["shares"][idx]
                pf_daily["total_cost"][pf_daily["code"] == value] += tmp_position["shares"][idx] * tmp_position["price"][idx]
                pf_daily["price_avg"][pf_daily["code"] == value] = pf_daily["total_cost"][pf_daily["code"] == value] / \
                                                                   pf_daily["shares"][pf_daily["code"] == value]
                print(i)
                print(value, "추가매수")
                # print(pf_daily)
            # not exists on holdings
            else:
                pf_daily = cbr(pf_daily, dataframe([[value, tmp_position["shares"][idx],
                                                     tmp_position["price"][idx],
                                                     tmp_position["shares"][idx]*tmp_position["price"][idx]]], columns=pf_daily.columns))
                print(i)
                print(value, "신규매수")
                # print(pf_daily)
        # update short position
        tmp_position = tmp_history[tmp_history["type"] == "매도"]
        for idx, value in enumerate(tmp_position["code"]):
            # break
            # already exits on holdings
            if value in pf_daily["code"].values:
                tmp_closed = {"date": i-timedelta(days=1), "code": tmp_position["code"][idx], "price_cost": pf_daily["price_avg"][pf_daily["code"] == value].iloc[0],
                              "price_sell": tmp_position["price"][idx], "shares_sell": tmp_position["shares"][idx],
                              "return": round((tmp_position["price"][idx] - pf_daily["price_avg"][pf_daily["code"] == value]).iloc[0] / pf_daily["price_avg"][pf_daily["code"] == value].iloc[0], 3)}
                tmp_closed["profit"] = (tmp_closed["price_sell"] - tmp_closed["price_cost"]) * tmp_closed["shares_sell"]
                table_closed = table_closed.append(tmp_closed, ignore_index=True)
                pf_daily["shares"][pf_daily["code"] == value] -= tmp_position["shares"][idx]
                # if the stock is sold out
                if pf_daily["shares"][pf_daily["code"] == value].iloc[0] == 0:
                    pf_daily = pf_daily[~(pf_daily["code"] == value)]
                    pf_daily.reset_index(drop=True, inplace=True)
                # if not sold out, update average price
                else:
                    pf_daily["total_cost"][pf_daily["code"] == value] -= pf_daily["price_avg"][pf_daily["code"] == value] * tmp_position["shares"][idx]
                print(i)
                print(value, "매도")
                # print(pf_daily)
            # not exists on holdings
            else:
                print("ERROR : not exists on holdings -> pass this operation")
                print(i)
                print(value, "공매도")
                # print(pf_daily)
                raise StopIteration

    # if market is not open, pass
    if i not in table_stockprice.index:
        continue
    # if market is open, calculate return on stocks
    else:
        date_prev = table_stockprice.index[findIdx(table_stockprice.index, [i])[0] - 1]
        price = [table_stockprice[corp_code][i] for corp_code in pf_daily["code"]]
        price_prev = [table_stockprice[corp_code][date_prev] for corp_code in pf_daily["code"]]
        market_value = price * pf_daily["shares"].values
        market_value_prev = price_prev * pf_daily["shares"].values
        price_return = [table_stockreturn[corp_code][i] for corp_code in pf_daily["code"]]

        # calculate the rate of change on the market value (including cash flow effect)
        tmp_pf_marketvalue.append(sum(market_value))
        # calculate the weighted return on each stocks (excluding cash flow effect)
        price_weighted_return = [value * (market_value_prev[idx] / sum(market_value_prev)) for idx, value in enumerate(price_return)]
        tmp_price_weighted_return.append(sum(price_weighted_return))

        pf_daily["sector"] = table_stock_info["sector"].iloc[[findIdx(table_stock_info["code"].values, [j])[0] for j in pf_daily["code"].values]].to_list()
        pf_daily["market_value"] = market_value
        pf_daily["price_weighted_return"] = price_weighted_return

        tmp_groupby_sector = pf_daily.groupby("sector").sum()
        for name in sector_names:
            if name not in tmp_groupby_sector.index.values:
                tmp_groupby_sector = tmp_groupby_sector.append(dataframe([[nan] * tmp_groupby_sector.shape[1]], index=[name], columns=tmp_groupby_sector.columns))
        tmp_groupby_sector = tmp_groupby_sector.loc[orderElems(tmp_groupby_sector.index, sector_names)]

        tmp_groupby_sector.reset_index(inplace=True)
        tmp_groupby_sector.index = [i] * len(sector_names)
        tmp_groupby_sector["weight"] = (tmp_groupby_sector["market_value"] / tmp_groupby_sector["market_value"].sum()).values
        table_sector = table_sector.append(tmp_groupby_sector[["index", "market_value", "weight", "price_weighted_return"]])

        if i != table_stockprice.index[-1]:
            pf_daily.drop(["sector", "market_value", "price_weighted_return"], axis=1, inplace=True)
        else:
            pf_daily["price_market"] = price
            pf_daily["profit"] = (pf_daily["price_market"] - pf_daily["price_avg"]) * pf_daily["shares"]
            pf_daily["total_return"] = (pf_daily["price_market"] - pf_daily["price_avg"]) / pf_daily["price_avg"]
            pf_daily["name"] = [table_stock_info["name"][table_stock_info["code"] == j].iloc[0] for j in pf_daily["code"]]
            pf_daily["weight"] = pf_daily["market_value"] / pf_daily["market_value"].sum()
            pf_daily = pf_daily[["code", "name", "sector", "shares", "price_avg", "total_cost",
                                 "price_market", "market_value", "total_return", "profit", "weight"]]
            pf_daily.sort_values(["sector", "weight"], ascending=[False, False], inplace=True)
            pf_daily.reset_index(drop=True, inplace=True)

print(pf_daily)
pf_daily.to_csv("projects/portfolio_visualization/portfolio_current.csv", index=False)
pf_daily.to_csv("projects/portfolio_visualization/portfolio_current_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=False)


# portfolio table
table_compare["portfolio"] = tmp_pf_marketvalue[1:]
table_compare["benchmark"] = table_stockprice["^GSPC"]

table_compare["portfolio_marketvalue_ror"] = series(tmp_pf_marketvalue).pct_change()[1:].values
table_compare["benchmark_marketvalue_ror"] = table_stockreturn["^GSPC"]

table_compare["portfolio_weighted_return"] = tmp_price_weighted_return
table_compare["benchmark_weighted_return"] = table_stockreturn["^GSPC"]
table_compare["portfolio_weighted_return_cumsum"] = table_compare["portfolio_weighted_return"].cumsum()
table_compare["benchmark_weighted_return_cumsum"] = table_compare["benchmark_weighted_return"].cumsum()

table_compare["portfolio_std"] = table_compare["portfolio_weighted_return"].rolling(20).std()
table_compare["benchmark_std"] = table_compare["benchmark_weighted_return"].rolling(20).std()
table_compare["portfolio_return_stdAdj"] = (table_compare["portfolio_weighted_return"] / table_compare["portfolio_std"])
table_compare["benchmark_return_stdAdj"] = (table_compare["benchmark_weighted_return"] / table_compare["benchmark_std"])
table_compare["portfolio_return_stdAdj_cumsum"] = table_compare["portfolio_return_stdAdj"].cumsum()
table_compare["benchmark_return_stdAdj_cumsum"] = table_compare["benchmark_return_stdAdj"].cumsum()

from numpy_ext import rolling_apply
table_compare["portfolio_beta"] = list(np.abs(rolling_apply(lambda x, y: stats.linregress(x, y)[0], 20,
                                                            table_compare["benchmark_weighted_return"], table_compare["portfolio_weighted_return"])))
table_compare["portfolio_return_betaAdj"] = (table_compare["portfolio_weighted_return"] / table_compare["portfolio_beta"])
table_compare["portfolio_return_betaAdj_cumsum"] = table_compare["portfolio_return_betaAdj"].cumsum()
# table_compare.fillna(0.0, inplace=True)

table_compare.head(10)
table_compare.head(30)
table_compare.tail(10)


# sector table
table_sector.columns = ["sector", "market_value", "weight", "return"]
table_sector.index.name = "date"
# table_sector.fillna(0.0, inplace=True)

# export compare table
table_compare.to_csv("projects/portfolio_visualization/portfolio_summary.csv", index=True)
table_compare.to_csv("projects/portfolio_visualization/portfolio_summary_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=True)

table_compare_recent_stats = dataframe()
table_compare_recent_stats = table_compare.iloc[-1]
table_compare_recent_stats = table_compare_recent_stats.to_frame().T.reset_index(drop=True)
table_compare_recent_stats.to_csv("projects/portfolio_visualization/portfolio_stats.csv", index=False)
table_compare_recent_stats.to_csv("projects/portfolio_visualization/portfolio_stats_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=False)


# export sector table
table_sector.to_csv("projects/portfolio_visualization/sector_summary.csv", index=True)
table_sector.to_csv("projects/portfolio_visualization/sector_summary_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=True)

# export sector return and weight
table_sector.groupby("sector").sum()

table_sector_return = pd.pivot_table(table_sector.reset_index(), index='sector', values='return', columns='date', aggfunc='sum').T
table_sector_return = table_sector_return[orderElems(table_sector_return.columns, sector_names)]
table_sector_return.to_csv("projects/portfolio_visualization/sector_summary_return.csv", index=True)
table_sector_return.to_csv("projects/portfolio_visualization/sector_summary_return_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=True)

table_sector_weight = pd.pivot_table(table_sector.reset_index(), index='sector', values='weight', columns='date', aggfunc='sum').T
table_sector_weight = table_sector_weight[orderElems(table_sector_weight.columns, sector_names)]
table_sector_weight.to_csv("projects/portfolio_visualization/sector_summary_weight.csv", index=True)
table_sector_weight.to_csv("projects/portfolio_visualization/sector_summary_weight_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=True)

table_sector_recent_stats = dataframe(index=sector_names)
table_sector_recent_stats["return_cumsum"] = table_sector_return.cumsum().iloc[-1]
table_sector_recent_stats["weight"] = table_sector_weight.iloc[-1]
table_sector_recent_stats.index.name = "sector"
table_sector_recent_stats = table_sector_recent_stats.reset_index()
table_sector_recent_stats.to_csv("projects/portfolio_visualization/sector_stats.csv", index=False)
table_sector_recent_stats.to_csv("projects/portfolio_visualization/sector_stats_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=False)


# export closed postion table
table_closed.set_index("date", inplace=True)
table_closed["name"] = [table_stock_info["name"][table_stock_info["code"] == j].iloc[0] for j in table_closed["code"]]
table_closed["sector"] = table_stock_info["sector"].iloc[[findIdx(table_stock_info["code"].values, [j])[0] for j in table_closed["code"].values]].to_list()
table_closed = table_closed[["code", "name", "sector", "shares_sell", "price_cost", "price_sell", "profit", "return"]]
table_closed.to_csv("projects/portfolio_visualization/closed_summary.csv", index=True)
table_closed.to_csv("projects/portfolio_visualization/closed_summary_" + datetime.now().strftime("%Y-%m-%d") + ".csv", index=True)

# total profit on closed position
print("USD", round(table_closed.sum()["profit"], 3))
# average return on closed position
print(round(table_closed.mean()["return"], 3) * 100, "%")


# ===== upload files to the google drive
# api key : AIzaSyDs85RBcOR2pWG__TfAvDKwn3HuXeRVzOo
from googleapiclient.http import MediaFileUpload
from google_drive_service import Create_Service

CLENT_SECRET = 'clent_secret_GoogleCloudDemp.json'
API_NAME = "drive"
API_VERSION = "v3"
SCOPES = ["https://www.googleapis.com/auth/drive"]

service = Create_Service(CLENT_SECRET, API_NAME, API_VERSION, SCOPES)

folder_id = "115HUT0LciOxDux3fLYCJt_N_poJQSxkm"
file_names = ["closed_summary_2022-02-01.csv"]
mime_types = ["text/csv"]

for file_name, mime_type in zip(file_names, mime_types):
    file_metadata = {
        "name": file_name,
        "parents": [folder_id]
    }
    media = MediaFileUpload(F"projects/portfolio_visualization/{filename}", mimetype=mime_type)
    service.files().create(
        body=file_metadata,
        media_body=mdia,
        fields="id"
    ).execute()


# file_metadata = {'name': 'closed_summary.csv'}
# media = MediaFileUpload('projects/portfolio_visualization/closed_summary.csv')
# file = drive_service.files().create(body=file_metadata,
#                                     media_body=media,
#                                     fields='id').execute()
# # print 'File ID: %s' % file.get('id')
