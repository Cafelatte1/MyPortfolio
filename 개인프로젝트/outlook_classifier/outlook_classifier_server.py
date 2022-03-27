# 메일 분류하기 실습
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
import win32com.client as client
import pathlib
import rhinoMorph

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import random as tf_rnd
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

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
                if pd.Timestamp(j.ReceivedTime).tz_convert(None) < cond_time:
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

# class_names = [i.Name for i in inbox.Folders if i.Name not in ("@classifier", "test", "temp", "tips")]
# # print(class_names)
# class_map = dataframe()
# class_map["class"] = class_names
# class_map["class_mapped"] = list(range(1, len(class_names) + 1))
# class_map.to_csv(os.path.join(local_path, "class_mapped.csv"), index=False)
class_map = read_csv(os.path.join(local_path, "class_mapped.csv"))
class_map["target_folder"] = class_map["class"]
class_map["target_folder"][-1:] = "@classifier"

# easyIO(class_map, "C:/Users/flash/PycharmProjects/pythonProject/projects/outlook_practice/class_mapped.pickle", "w")
# class_map = easyIO(None, "C:/Users/flash/PycharmProjects/pythonProject/projects/outlook_practice/class_mapped.pickle", "r")


text_finder_subject = re.compile('[^ _/A-Za-zㄱ-ㅣ가-힣+]')
text_finder_body = re.compile('[^ _/A-Za-zㄱ-ㅣ가-힣+]')
text_finder_atts = re.compile('[^ ./A-Za-zㄱ-ㅣ가-힣+]')
text_finder = re.compile('[^;A-Za-zㄱ-ㅣ가-힣+]')
rn = rhinoMorph.startRhino()
# inbox_dic = dict.fromkeys(class_map["class"])
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
# for idx, value in enumerate(tqdm(class_map["class"])):
#     # 해당 폴더 및 하위 폴더의 모든 메일을 dataframe 으로 저장 리턴하는 함수 호출
#     inbox_dic[value] = get_all_df(inbox.Folders[value])
#     # 생서된 dataframe에 클래스 할당
#     inbox_dic[value]["class"] = class_map["class_mapped"].iloc[idx]


# cond_time = datetime.now() - DateOffset(days=3)
# # 기본 메일함인 '받은 편지함'에 대한 dataframe 생성 및 클래스 0번으로 할당
# # inbox_dic["받은 편지함"] = get_all_emails(inbox.Items, cond_time)
target_folder = inbox.Folders["@classifier"]
full_df = get_all_emails(target_folder.Items)
full_df["class"] = -1

# full_df = dataframe()
# for i in inbox_dic.values():
#     full_df = full_df.append(i)
# full_df.reset_index(drop=True, inplace=True)


full_df["doc"] = full_df["subject"] + " " + full_df["body"] + " " + full_df["atts"].apply(lambda x: "" if x == "none" else x)
# full_df["viewer"] = full_df["to"] + " " + full_df["cc"]
full_df["viewer"] = full_df["sender_name"] + " " + full_df["to"] + " " + full_df["cc"]

# full_df = full_df[["sender_name", "viewer", "doc", "hour", "week_of_month", "class"]]
full_df = full_df[["viewer", "doc", "hour", "week_of_month", "class"]]
full_y = full_df[["class"]].astype("int32")
print(full_df.shape)
num_items = full_df.shape[0]
modelsave_filepath = os.path.join(local_path, pathlib.WindowsPath("models/"))
checkpoint_filepath = os.path.join(modelsave_filepath, pathlib.WindowsPath("checkpoint/"))
# if os.path.exists(modelsave_filepath):
#     shutil.rmtree(modelsave_filepath)
# createFolder(modelsave_filepath)
# createFolder(checkpoint_filepath)

max_vocab = 1024
doc_vectorize = load_model(os.path.join(local_path, "models/doc_vectorizer"))
doc_embed = doc_vectorize(full_df["doc"].to_numpy()).numpy()

viewer_vectorize = load_model(os.path.join(local_path, "models/viewer_vectorizer"))
viewer_embed = viewer_vectorize(full_df["viewer"].to_numpy()).numpy()
viewer_embed = pad_sequences(viewer_embed, 4, padding='post', truncating='post', value=0)

full_x = full_df.drop(["doc", "viewer", "class"], axis=1)
del full_df


# learning parameter setting
epochs = 100
patient_epochs = int(epochs * 0.2)
patient_lr = int(patient_epochs * 0.2)
eta = 1e-3
weight_decay = 1e-4
model_save_flag = True
batch_size = 8
nfolds = 5
prob_df = np.zeros((full_x.shape[0], len(class_map)))

for fold in range(nfolds):
    tmp_time = time()

    print("\n===== Fold", fold, "=====\n")

    tmp_df = (
        (
            # full_x[["sender_name"]].to_numpy(),
            full_x[["week_of_month"]].to_numpy(),
            full_x[["hour"]].to_numpy(),
            viewer_embed,
            doc_embed
        ), None
    )
    test_ds = tf.data.Dataset.from_tensor_slices(tmp_df).batch(batch_size).prefetch(2)

    model = load_model(os.path.join(modelsave_filepath, pathlib.WindowsPath("fold_" + str(fold))))
    prob_df += model.predict(test_ds) / nfolds

    tf.keras.backend.clear_session()
    print("Fold " + str(fold) + " Time (minutes) : ", round((time() - tmp_time) / 60, 3))

pred_df = prob_df.argmax(axis=1)

move_folders = []
for folder_class, item in zip(pred_df, target_folder.Items):
    folder_name = class_map["target_folder"][class_map["class_mapped"] == folder_class].iloc[0]
    if folder_name != "@classifier":
        print(item.Subject, "\n  is moved to", folder_name, "\n")
        move_folders.append(folder_name)
        # if not item.Unread:
        #     item.GetConversation().MarkAsUnread()
        # item.Move(inbox.Folders[folder_name])
    else:
        print(item.Subject, "\n  is normal mail\n")
        move_folders.append("normal")
        # if not item.Unread:
        #     item.GetConversation().MarkAsUnread()
        # item.Move(inbox)

for idx in range(num_items):
    folder_name = move_folders[idx]
    if folder_name == "normal":
        print(target_folder.Items[0].Subject, "\n  is moved to", folder_name, "\n")
        if not target_folder.Items[0].Unread:
            target_folder.Items[0].GetConversation().MarkAsUnread()
        target_folder.Items[0].Move(target_folder.Folders["normal"])
        # inbox.Folders["@classifier"]
    else:
        print(target_folder.Items[0].Subject, "\n  is moved to", folder_name, "\n")
        if not target_folder.Items[0].Unread:
            target_folder.Items[0].GetConversation().MarkAsUnread()
        target_folder.Items[0].Move(inbox.Folders[folder_name])
    sleep(0.25)