import os
import pathlib
import pandas as pd
import torch
import signatory

from spdlog import ConsoleLogger

MAXTIMEINICU = 72
SIG_DEPTH = 2

logger = ConsoleLogger("preprocess.py", multithreaded=True, stdout=True, colored=True)

here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsis'
preprocessed_loc = base_loc / 'preprocessed'


def signme(df):
    return torch.rand(1, 6)


if not os.path.exists(preprocessed_loc):
    os.mkdir(preprocessed_loc)

alldata = pd.DataFrame()

psv_list = []
psv_glob = base_loc.glob("*.psv")

for psv_file in sorted(psv_glob):
    logger.info(f'Loading file {psv_file}')
    df = pd.read_csv(psv_file, sep='|')
    df = df.drop(df[df.ICULOS > MAXTIMEINICU].index)
    alldata = pd.concat((alldata, df))

allmeans = alldata.mean()

psv_glob = base_loc.glob("*.psv")
for psv_file in sorted(psv_glob):
    logger.info(f'Processing file {psv_file}')
    df = pd.read_csv(psv_file, sep='|')
    df = df.drop(df[df.ICULOS > MAXTIMEINICU].index)

    for (column_name, column_data) in df.iteritems():
        df[column_name] = df[column_name].fillna(allmeans[column_name])

    sepsis_label = max(df["SepsisLabel"])

    processed_file = (preprocessed_loc / psv_file.stem).with_suffix(".csv.bz2")

    out_df = pd.DataFrame()
    for i in df["ICULOS"]:
        tmp_df = df.drop(df[df.ICULOS > i].index)
        label_t = torch.tensor([(sepsis_label, i)])
        signed_t = signatory.signature(tmp_df, SIG_DEPTH)

        data_df = pd.DataFrame(torch.cat((label_t, signed_t), 1).numpy())
        out_df = pd.concat((out_df, data_df))

    logger.info(f'Writing processed output file to {processed_file}')
    out_df.to_csv(processed_file, index=False, header=False, compression='bz2')
