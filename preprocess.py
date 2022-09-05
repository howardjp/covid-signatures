import json
import os
import pathlib
import pandas as pd
import pickle
import torch
import signatory

from spdlog import ConsoleLogger

MAXTIMEINICU = 72
SIG_DEPTH = 2
NAN_THRESH = 0.98

logger = ConsoleLogger("preprocess.py", multithreaded=True, stdout=True, colored=True)

here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsistest'
#trnA_loc = base_loc / 'training'
preprocessed_loc = base_loc / 'preprocessed'
allmeans_file = base_loc / 'allmeans.pickle'
nanrate_file = base_loc / 'dropcols.pickle'

def signme(df):
    # return torch.rand(1, 6)
    return signatory.signature(df, SIG_DEPTH)


if not os.path.exists(preprocessed_loc):
    os.mkdir(preprocessed_loc)

alldata = pd.DataFrame()

psv_list = []
psv_glob = trnA_loc.glob("*.psv")

allmeans = pd.DataFrame()
dropcols = []

if os.path.exists(allmeans_file):
    logger.info(f'Loading column means from {allmeans_file}')
    with open(allmeans_file, 'rb') as f:
        allmeans = pickle.load(f)
    with open(nanrate_file, 'rb') as f:
        dropcols = pickle.load(f)
else:
    logger.info(f'Column means file {allmeans_file} not found')
    for psv_file in sorted(psv_glob):
        logger.info(f'Loading file {psv_file}')
        df = pd.read_csv(psv_file, sep='|')
        df = df.drop(df[df.ICULOS > MAXTIMEINICU].index)
        alldata = pd.concat((alldata, df))

    for (column_name, column_data) in alldata.iteritems():
        nanrate = alldata[column_name].isna().sum() / alldata.shape[0]
        if nanrate >= NAN_THRESH:
            logger.info(f'Column {column_name} removed for {nanrate} missing values')
            dropcols.append(column_name)

    allmeans = alldata.mean()
    with open(allmeans_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(allmeans, f, pickle.HIGHEST_PROTOCOL)
    with open(nanrate_file, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(dropcols, f, pickle.HIGHEST_PROTOCOL)

psv_glob = trnA_loc.glob("*.psv")
for psv_file in sorted(psv_glob):
    logger.info(f'Processing file {psv_file}')
    df = pd.read_csv(psv_file, sep='|')

    min_time = min(df["ICULOS"])
    df["ICULOS"] = df["ICULOS"] - min_time
    df = df.drop(df[df.ICULOS > MAXTIMEINICU].index)

    for (column_name, column_data) in df.iteritems():
        df[column_name] = df[column_name].fillna(allmeans[column_name])

    sepsis_label = max(df["SepsisLabel"])

    processed_file = (preprocessed_loc / psv_file.stem).with_suffix(".csv")

    time_index = list(df["ICULOS"])
    time_index.remove(min(time_index))

    times_s = df.drop(df[df.SepsisLabel == 1].index).ICULOS
    if len(times_s) != 0:
        max_time = max(times_s)
        df = df.drop(df[df.ICULOS > max_time].index)

        time_index = list(df["ICULOS"])
        time_index.remove(min(time_index))

        phistory = torch.empty(0)

        p_age = max(df['Age'])
        p_sex = max(df['Gender'])

        df = df.drop(columns='SepsisLabel')
        df = df.drop(columns='HospAdmTime')
        df = df.drop(columns='Unit1')
        df = df.drop(columns='Unit2')
        df = df.drop(columns='Age')
        df = df.drop(columns='Gender')
        df = df.drop(columns=dropcols)

        if len(time_index) != 0:
            for i in time_index:
                tmp_df = df.drop(df[df.ICULOS > i].index)
                tmp_df = tmp_df.drop(columns='ICULOS')
                tmp_t = torch.tensor([tmp_df.values], dtype=torch.float)
                tmp_sign = signatory.signature(tmp_t, SIG_DEPTH)
                tmp_label = torch.tensor([[sepsis_label, max_time - i, p_age, p_sex]])
                tmp_row = torch.cat((tmp_label, tmp_sign), dim=1)
                phistory = torch.cat((phistory, tmp_row), dim=0)

            logger.info(f'Writing processed output file of size {phistory.size()} to {processed_file}')
            with open(processed_file, 'w', encoding='utf-8') as f:
                pd.DataFrame(phistory.numpy()).to_csv(f, header=False, index=False)

