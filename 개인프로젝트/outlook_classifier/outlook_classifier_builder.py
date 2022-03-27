# mail classifier
import numpy as np
from numpy import array, nan, random as np_rnd, where
import pandas as pd
from pandas import DataFrame as dataframe, Series as series, isna, read_csv
from pandas.tseries.offsets import DateOffset

import os
from multiprocessing import cpu_count
import copy
import pickle
import warnings
from datetime import datetime, timedelta
from time import time, sleep, mktime
from tqdm import tqdm
import re
import random as rnd
from sklearn.model_selection import StratifiedKFold

import win32com.client as client
import pathlib
import rhinoMorph
import shutil

import tensorflow as tf
from tensorflow import random as tf_rnd
from tensorflow.data import Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import callbacks as tf_callbacks
from tqdm.keras import TqdmCallback
import tensorflow_addons as tfa
import tensorflow_recommenders as tfrs
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras import losses as tf_losses
from tensorflow.keras import metrics as tf_metrics


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


# font_path = 'myfonts/NanumSquareB.ttf'
# font_obj = fm.FontProperties(fname=font_path, size=12).get_name()
# rc('font', family=font_obj)

# 3  Deleted Items
# 4  Outbox
# 5  Sent Items
# 6  Inbox
# 9  Calendar
# 10 Contacts
# 11 Journal
# 12 Notes
# 13 Tasks
# 14 Drafts

def text_extractor(string, lang="eng", spacing=True):
    # # 괄호를 포함한 괄호 안 문자 제거 정규식
    # re.sub(r'\([^)]*\)', '', remove_text)
    # # <>를 포함한 <> 안 문자 제거 정규식
    # re.sub(r'\<[^)]*\>', '', remove_text)
    if lang == "eng":
        text_finder = re.compile('[^ /A-Za-z]') if spacing else re.compile('[^/A-Za-z]')
    elif lang == "kor":
        text_finder = re.compile('[^ /ㄱ-ㅣ가-힣+]') if spacing else re.compile('[^/ㄱ-ㅣ가-힣+]')
    elif lang == "both":
        text_finder = re.compile('[^ /A-Za-zㄱ-ㅣ가-힣+]') if spacing else re.compile('[^/A-Za-zㄱ-ㅣ가-힣+]')
    else:
        text_finder = re.compile('[^ /0-9A-Za-zㄱ-ㅣ가-힣+]') if spacing else re.compile('[^/0-9A-Za-zㄱ-ㅣ가-힣+]')
    return text_finder.sub('', string)
def easyIO(x=None, path=None, op="r", keras_inspection=False):
    tmp = None
    if op == "r":
        with open(path, "rb") as f:
            tmp = pickle.load(f)
        return tmp
    elif op == "w":
        print(x)
        tmp = x
        if keras_inspection:
            tmp = {}
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
def createFolder(directory):
    try:
        if not os.path.exists(pathlib.WindowsPath(directory)):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    rnd.seed(seed)
    np_rnd.seed(seed)
    tf_rnd.set_seed(seed)
def get_all_emails(sub_item, cond_time=None):
    if sub_item.Count > 0:
        inbox_dic = pd.DataFrame()
        tmp_subject = []
        tmp_sender_name = []
        tmp_to = []
        tmp_cc = []
        tmp_body = []
        tmp_received_time = []
        tmp_atts = []
        tmp_atts_num = []
        for j in sub_item:

            if cond_time is not None:
                if not cond_time[1] >= pd.Timestamp(j.ReceivedTime).tz_convert(None) >= cond_time[0]:
                    continue

            try:
                tmp = text_finder_subject.sub('', j.Subject).replace("/", " ").replace("_", " ").lower()
                tmp = tmp.replace("ycadfid", "")
                tmp = tmp[2:] if tmp[:2] in ("re", "fw") else tmp
                tmp = tmp.split()

                tmp_stopwords_removed = []
                flag = True
                for z in tmp:
                    for k in body_stopwords:
                        if k in z:
                            flag = False
                            break
                    if flag:
                        tmp_stopwords_removed.append(z)
                    else:
                        flag = True
                tmp_stopwords_removed = " ".join(tmp_stopwords_removed)

                tmp_subject.append(" ".join([z.lower() for z in rhinoMorph.onlyMorph_list(rn, tmp_stopwords_removed) if len(z) > 1]))
                # print("complete: subject")
            except:
                print("except: subject")
                tmp_subject.append("unknown")

            try:
                tmp_sender_name.append(text_finder.sub('', j.SenderName).lower())
                # print("complete: sender_name")
            except:
                tmp_sender_name.append("unknown")
                print("except: sender_name")

            try:
                tmp_to.append(text_finder.sub('', j.To).lower().replace(";", " "))
                # print("complete: to")
            except:
                tmp_to.append("unknown")
                print("except: to")

            try:
                # inbox_dic[i]["to"][0] == ["hongkihyeon"]
                tmp_cc.append(text_finder.sub('', j.CC).lower().replace(";", " "))
                # print("complete: cc")
            except:
                tmp_cc.append("unknown")
                print("except: cc")

            try:
                tmp = [z for z in text_finder_body.sub('', j.Body).replace("/", " ").replace("_", " ").lower().split()]

                tmp_stopwords_removed = []
                flag = True
                for z in tmp:
                    for k in body_stopwords:
                        if k in z:
                            flag = False
                            break
                    if flag:
                        tmp_stopwords_removed.append(z)
                    else:
                        flag = True
                tmp_stopwords_removed = " ".join(tmp_stopwords_removed)

                tmp_body.append(" ".join([z.lower() for z in rhinoMorph.onlyMorph_list(rn, tmp_stopwords_removed) if len(z) > 1]))
                # print("complete: body")
            except:
                tmp_body.append("unknown")
                print("except: body")

            try:
                tmp_received_time.append(pd.Timestamp(j.ReceivedTime).tz_convert(None))
            except:
                tmp_body.append(pd.Timestamp(year=1970, month=1, day=1))
                print("except: received_time")

            try:
                tmp = j.attachments
                tmp_atts_num.append(tmp.Count)
                if tmp_atts_num[-1] > 0:
                    # tmp.Item(2).FileName
                    tmp_str = " ".join([text_finder_atts.sub('', tmp.Item(z).FileName.lower()) for z in range(1, tmp.Count + 1)])
                    tmp_atts.append(" ".join([z.lower() for z in rhinoMorph.onlyMorph_list(rn, tmp_str) if len(z) > 1]))
                else:
                    tmp_atts.append("none")
            except:
                tmp_atts.append("none")
                print("except: atts")

        inbox_dic["subject"] = tmp_subject
        inbox_dic["subject"].apply(lambda x: x[2:] if x[:2] in ("re", "fw") else x)
        inbox_dic["sender_name"] = tmp_sender_name
        inbox_dic["to"] = tmp_to
        inbox_dic["cc"] = tmp_cc
        inbox_dic["body"] = tmp_body
        inbox_dic["received_time"] = tmp_received_time
        inbox_dic["hour"] = inbox_dic["received_time"].dt.hour.astype("int32")
        inbox_dic["week_of_month"] = inbox_dic["received_time"].apply(week_of_month).astype("int32")
        inbox_dic["atts"] = tmp_atts
        inbox_dic["atts_num"] = tmp_atts_num
        return inbox_dic
def get_all_df(class_folder):
    return_df = dataframe()
    if class_folder.Folders.Count < 1:
        return return_df.append(get_all_emails(class_folder.Items))
    else:
        for i in class_folder.Folders:
            return_df = return_df.append(get_all_df(i))
        return return_df.append(get_all_emails(class_folder.Items))
def week_of_month(date):
    month = date.month
    week = 0
    while date.month == month:
        week += 1
        date -= timedelta(days=7)
    return week

local_path = os.path.join(os.environ['LOCALAPPDATA'], pathlib.WindowsPath("outlook_classifier/"))
createFolder(local_path)

outlook = client.Dispatch("Outlook.Application")
namespace = outlook.GetNameSpace("MAPI")
# # 초안 메일함 로드
# drafts = namespace.GetDefaultFolder(16)
# 메일함 로드
root_box = namespace.Folders["s_youngjunkim@yulchon.com"]
inbox = root_box.Folders["받은 편지함"]

class_names = [i.Name for i in inbox.Folders if i.Name not in ("@classifier", "test", "temp", "tips")]
# print(class_names)
class_map = dataframe()
class_map["class"] = class_names
class_map["class_mapped"] = list(range(1, len(class_names)+1))

# easyIO(class_map, "C:/Users/flash/PycharmProjects/pythonProject/projects/outlook_practice/class_mapped.pickle", "w")

# class_map = easyIO(None, "C:/Users/flash/PycharmProjects/pythonProject/projects/outlook_practice/class_mapped.pickle", "r")

text_finder_subject = re.compile('[^ _/A-Za-zㄱ-ㅣ가-힣+]')
text_finder_body = re.compile('[^ _/A-Za-zㄱ-ㅣ가-힣+]')
text_finder_atts = re.compile('[^ ./A-Za-zㄱ-ㅣ가-힣+]')
text_finder = re.compile('[^;A-Za-zㄱ-ㅣ가-힣+]')
rn = rhinoMorph.startRhino()
inbox_dic = dict.fromkeys(class_map["class"])
body_stopwords = [
    "iwl", "yulchoncom", "from", "sent", "pmto", "file", "vs", "re", "fw", "gen", "x", "subject", "yccloud", "command",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"
]

# test_df = get_all_df(inbox.Folders["Z드라이브 - 정기"])
# print(test_df.shape)
# print(test_df.shape[0] == 61 + 66 + 8)
# print(test_df.head())

# 폴더이름 별로 반복
for idx, value in enumerate(tqdm(class_map["class"])):
    # 해당 폴더 및 하위 폴더의 모든 메일을 dataframe 으로 저장 리턴하는 함수 호출
    inbox_dic[value] = get_all_df(inbox.Folders[value])
    # 생서된 dataframe에 클래스 할당
    inbox_dic[value]["class"] = class_map["class_mapped"].iloc[idx]

cond_time = (
    (datetime.now() - DateOffset(days=7)).normalize(),
    pd.to_datetime(datetime.now()).normalize() - DateOffset(seconds=1)
)

# 기본 메일함인 '받은 편지함'에 대한 dataframe 생성 및 클래스 0번으로 할당
inbox_dic["받은 편지함"] = get_all_emails(inbox.Items, cond_time)
inbox_dic["받은 편지함"]["class"] = 0
class_map = class_map.append(dataframe([["받은 편지함", 0]], columns=class_map.columns))
class_map.to_csv(os.path.join(local_path, "class_mapped.csv"), index=False)

full_df = dataframe()
for i in inbox_dic.values():
    full_df = full_df.append(i)
full_df.reset_index(drop=True, inplace=True)

full_df["doc"] = full_df["subject"] + " " + full_df["body"] + " " + full_df["atts"].apply(lambda x: "" if x == "none" else x)
# full_df["viewer"] = full_df["to"] + " " + full_df["cc"]
full_df["viewer"] = full_df["sender_name"] + " " + full_df["to"] + " " + full_df["cc"]


# full_df = full_df[["sender_name", "viewer", "doc", "hour", "week_of_month", "class"]]
full_df = full_df[["viewer", "doc", "hour", "week_of_month", "class"]]
full_y = full_df[["class"]].astype("int32")
print(full_df.shape)


def create_model_vectorization(input_shape, max_vocab, adapt_corpus, output_mode="int", output_sequence_length=None):
    vectorizer = layers.TextVectorization(
            max_tokens=max_vocab,
            output_mode=output_mode,
            output_sequence_length=output_sequence_length
        )
    vectorizer.adapt(adapt_corpus)
    model = Sequential([
        layers.InputLayer(input_shape=input_shape, dtype=tf.string),
        vectorizer
    ])
    return model

modelsave_filepath = os.path.join(local_path, pathlib.WindowsPath("models/"))
checkpoint_filepath = os.path.join(modelsave_filepath, pathlib.WindowsPath("checkpoint/"))
if os.path.exists(modelsave_filepath):
    shutil.rmtree(modelsave_filepath)
createFolder(modelsave_filepath)
createFolder(checkpoint_filepath)

max_vocab = 1024
doc_vectorize = create_model_vectorization(1, max_vocab, full_df["doc"].to_numpy(), output_mode="tf_idf")
doc_vectorize.compile()
doc_embed = doc_vectorize(full_df["doc"].to_numpy()).numpy()
doc_vectorize.save(os.path.join(local_path, "models/doc_vectorizer"))

viewer_vectorize = create_model_vectorization(1, int(max_vocab / 2), full_df["viewer"].to_numpy(), output_mode="int")
viewer_vectorize.compile()
viewer_embed = viewer_vectorize(full_df["viewer"].to_numpy()).numpy()
viewer_embed = pad_sequences(viewer_embed, 4, padding='post', truncating='post', value=0)
viewer_vectorize.save(os.path.join(local_path, "models/viewer_vectorizer"))

full_x = full_df.drop(["doc", "viewer", "class"], axis=1)
del full_df

def create_model():
    input_list = []
    concat_list = []

    dcn_list = []
    for i in full_x:
        if full_x[i].dtype.name in ("object"):
            input_list.append(layers.Input(shape=1, dtype=tf.string))
            x = layers.StringLookup(vocabulary=full_x[i].unique())(input_list[-1])
            dcn_list.append(layers.Embedding(input_dim=len(full_x[i].unique()) + 1, output_dim=4, embeddings_initializer="glorot_normal")(x))
        elif full_x[i].dtype.name in ("int32", "int64"):
            input_list.append(layers.Input(shape=1, dtype=tf.int32))
            x = layers.IntegerLookup(vocabulary=full_x[i].unique())(input_list[-1])
            dcn_list.append(layers.Embedding(input_dim=len(full_x[i].unique()) + 1, output_dim=4)(x))

    input_list.append(layers.Input(shape=4))
    x = layers.Embedding(input_dim=int(max_vocab / 2), output_dim=32, embeddings_initializer="glorot_normal")(input_list[-1])
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(4)(x)
    dcn_list.append(layers.GlobalMaxPool1D(keepdims=True)(x))

    # DCN(Deep Cross Network for feature interaction)
    x = layers.Concatenate(axis=1)(dcn_list)
    dcn1 = tfrs.layers.dcn.Cross()(x, layers.Dropout(0.25)(x))
    dcn2 = tfrs.layers.dcn.Cross()(x, layers.Dropout(0.25)(dcn1))

    concat_list.append(layers.Flatten()(x))
    concat_list.append(layers.Flatten()(dcn1))
    concat_list.append(layers.Flatten()(dcn2))

    input_list.append(layers.Input(shape=max_vocab, dtype=tf.float32))
    x = layers.Dropout(0.5)(input_list[-1])
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    concat_list.append(layers.Dense(128, activation="relu")(x))

    x = layers.Concatenate()(concat_list)
    block1_input = layers.Dropout(0.25)(x)
    x = tfa.layers.WeightNormalization(
        layers.Dense(64, activation="relu")
    )(block1_input)
    x = layers.Dropout(0.25)(x)

    block2_input = layers.Concatenate()([block1_input, x])
    x = tfa.layers.WeightNormalization(
        layers.Dense(64, activation="relu")
    )(block2_input)
    x = layers.Dropout(0.25)(x)

    block3_input = layers.Concatenate()([block2_input, x])
    x = tfa.layers.WeightNormalization(
        layers.Dense(64, activation="relu")
    )(block3_input)
    x = layers.Dropout(0.25)(x)

    x = tfa.layers.WeightNormalization(
        layers.Dense(32, activation="swish")
    )(x)

    classifier = layers.Dense(len(class_map), activation="softmax")(x)
    return Model(input_list, classifier)
# create_model().summary()


# learning parameter setting
epochs = 100
patient_epochs = int(epochs * 0.2)
patient_lr = int(patient_epochs * 0.2)
eta = 1e-3
weight_decay = 1e-4

batch_size = 8
nfolds = 5
kfolds_spliter = StratifiedKFold(nfolds, shuffle=True, random_state=42)
model_save_flag = True

# fold training
seed_everything()
perf_table = dataframe()
for fold, (train_idx, val_idx) in enumerate(kfolds_spliter.split(range(full_x.shape[0]), full_y["class"].values)):
    tmp_time = time()
    
    print("\n===== Fold", fold, "=====\n")

    tmp_df = (
        (
            # full_x[["sender_name"]].iloc[train_idx].to_numpy(),
            full_x[["week_of_month"]].iloc[train_idx].to_numpy(),
            full_x[["hour"]].iloc[train_idx].to_numpy(),
            viewer_embed[train_idx],
            doc_embed[train_idx]
        ), full_y.iloc[train_idx].to_numpy()
    )
    train_ds = Dataset.from_tensor_slices(tmp_df).shuffle(int(batch_size / 2), reshuffle_each_iteration=True).cache().batch(batch_size).prefetch(2)
    tmp_df = (
        (
            # full_x[["sender_name"]].iloc[val_idx].to_numpy(),
            full_x[["week_of_month"]].iloc[val_idx].to_numpy(),
            full_x[["hour"]].iloc[val_idx].to_numpy(),
            viewer_embed[val_idx],
            doc_embed[val_idx]
        ), full_y.iloc[val_idx].to_numpy()
    )
    val_ds = Dataset.from_tensor_slices(tmp_df).cache().batch(batch_size).prefetch(2)

    checkpoint_datapath = os.path.join(checkpoint_filepath, pathlib.WindowsPath("tmp/"))
    cb_earlyStopping = tf_callbacks.EarlyStopping(patience=patient_epochs, monitor='val_acc', mode='max')
    cb_reduceLR = tf_callbacks.ReduceLROnPlateau(patience=patient_lr, factor=0.8, min_lr=1e-5)
    cb_modelsave = tf_callbacks.ModelCheckpoint(filepath=checkpoint_datapath, monitor='val_acc', mode='max', save_weights_only=True, save_best_only=True)

    model = create_model()
    model.compile(
        loss=tf_losses.SparseCategoricalCrossentropy(),
        optimizer=tfa.optimizers.AdamW(learning_rate=eta, weight_decay=weight_decay),
        metrics=[tf_metrics.SparseCategoricalAccuracy(name="acc")]
    )
    print("start training")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0,
              callbacks=[cb_reduceLR, cb_earlyStopping, cb_modelsave, TqdmCallback(verbose=0)])
    print("end training")
    model.load_weights(checkpoint_datapath)

    perf_table = perf_table.append([model.evaluate(val_ds, return_dict=True, verbose=1)])
    if model_save_flag:
        model.save(os.path.join(modelsave_filepath, pathlib.WindowsPath("fold_" + str(fold))))

    clear_session()
    print("Fold " + str(fold) + " Time (minutes) : ", round((time() - tmp_time) / 60, 3))

shutil.rmtree(checkpoint_filepath)

perf_table.columns = ["logloss", "accuracy"]
perf_table.index = ["fold_" + str(i) for i in range(nfolds)]
perf_table.loc["average"] = perf_table.mean().values
perf_table.loc["std"] = perf_table.std().values
perf_table.to_csv(os.path.join(local_path, "performance_table_" + str(datetime.now()).split()[0] + ".csv"), index=True)

