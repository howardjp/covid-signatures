import json
import cbor2
import numpy as np
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn_pandas import DataFrameMapper

from spdlog import ConsoleLogger

import torchtuples as tt
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv


MAXPATIENTS = 1
TRNTSTSPLIT = 0.2
TRNVALSPLIT = 0.8

if 'interactive_mode' in globals():
    if interactive_mode == True:
        os.environ["TORCHSEED"] = "1"
        os.environ["NPSEED"] = "1"
        os.environ["CLUSTERID"] = "1"
        os.environ["PROCID"] = "1"

logger = ConsoleLogger("coxtime.py", multithreaded=True, stdout=True, colored=True)

TORCHSEED = int(os.environ.get('TORCHSEED'))
logger.info(f'Setting torch.manual_seed to {TORCHSEED}')
torch.manual_seed(TORCHSEED)

NPSEED = int(os.environ.get('NPSEED'))
logger.info(f'Setting np.random.seed to {NPSEED}')
np.random.seed(NPSEED)

here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsis'
# trnA_loc = base_loc / 'training'
trnA_loc = base_loc
preprocessed_loc = base_loc / 'preprocessed'
merged_data_file = base_loc / 'merged.csv'

logger.info(f'Getting data from {merged_data_file}')
alldata = pd.read_csv(merged_data_file, header=None).rename(columns={0: "id", 1: "event", 2: "duration"})
logger.info(f'Done loading data')
iddata = alldata[["id", "event"]].groupby("id", as_index=False).max().astype({'event': int})

#_, iddata = train_test_split(iddata, test_size=0.01, stratify=iddata['event'])

iddata_tst, iddata_trn = train_test_split(iddata, test_size=TRNTSTSPLIT, stratify=iddata['event'])
iddata_val, iddata_trn = train_test_split(iddata_trn, test_size=TRNVALSPLIT, stratify=iddata_trn['event'])

logger.info(f'Using {iddata_trn.size} patients for training')
logger.info(f'Using {iddata_val.size} patients for validation')
logger.info(f'Using {iddata_tst.size} patients for testing')

alldata_trn = pd.merge(left=iddata_trn.drop(columns="event"), right=alldata, left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata, left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})
alldata_val = pd.merge(left=iddata_val.drop(columns="event"), right=alldata, left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

standardize = [([col], StandardScaler()) for col in range(3, 760)]

x_mapper = DataFrameMapper(standardize)
x_trn = x_mapper.fit_transform(alldata_trn).astype('float32')
x_val = x_mapper.transform(alldata_val).astype('float32')
x_tst = x_mapper.transform(alldata_tst).astype('float32')

labtrans = CoxTime.label_transform()
get_target = lambda df: (df['duration'].values, df['event'].values)
y_trn = labtrans.fit_transform(*get_target(alldata_trn))
y_val = labtrans.transform(*get_target(alldata_val))
val = tt.tuplefy(x_val, y_val)

durations_test, events_test = get_target(alldata_tst)
durations_all, events_all = get_target(alldata_tst)

in_features = x_trn.shape[1]
num_nodes = [64, 64]
batch_norm = True
dropout = 0.1
net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

batch_size = 256
lrfinder = model.lr_finder(x_trn, y_trn, batch_size, tolerance=2)
_ = lrfinder.plot()

learn_rate = min(lrfinder.get_best_lr(), 0.0001)

logger.info(f'Found learning rate of {lrfinder.get_best_lr()}, using learning rate of {learn_rate}')
model.optimizer.set_lr(learn_rate)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping(patience=25)]
verbose = True

log = model.fit(x_trn, y_trn, batch_size, epochs, callbacks, verbose, val_data=val)

results = dict()

results["torch_seed"] = TORCHSEED
results["np_seed"] = NPSEED
results["training_history"] = json.loads(log.to_pandas().to_json())

_ = log.plot()
_ = model.compute_baseline_hazards()
surv = model.predict_surv_df(x_tst)
surv.iloc[:, :5].plot()
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
logger.info(f'Found concordance_td: {ev.concordance_td()}')
time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
_ = ev.brier_score(time_grid).plot()

logger.info(f'Found integrated_brier_score: {ev.integrated_brier_score(time_grid)}')
logger.info(f'Found integrated_nbll: {ev.integrated_nbll(time_grid)}')

results["72hr"] = dict()
results["72hr"]['antolini'] = ev.concordance_td('antolini')
results["72hr"]['antoliniadj'] = ev.concordance_td('adj_antolini')
results["72hr"]['ibs'] = ev.integrated_brier_score(time_grid)
results["72hr"]['inbll'] = ev.integrated_nbll(time_grid)
results["72hr"]['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
results["72hr"]['nbll'] = ev.nbll(time_grid).to_numpy().tolist()
results["72hr"]['iloc'] = surv.iloc().obj.to_dict()
results["72hr"]['truth'] = events_test.tolist()

maxtimes_by_id = pd.DataFrame(alldata.groupby(['id'], sort=False)['duration'].max())
maxtimes_by_id.rename(columns={"id": "id", "duration": "maxtime"}, inplace=True)

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration != 24].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

durations_test, events_test = get_target(alldata_tst)
x_tst = x_mapper.transform(alldata_tst).astype('float32')
surv = model.predict_surv_df(x_tst)

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
logger.info(f'Found 24-hour concordance_td: {ev.concordance_td()}')
time_grid = np.linspace(durations_all.min(), durations_all.max(), 100)

logger.info(f'Found 24-hour integrated_brier_score: {ev.integrated_brier_score(time_grid)}')
logger.info(f'Found 24-hour integrated_nbll: {ev.integrated_nbll(time_grid)}')

results["24hr"] = dict()
results["24hr"]['antolini'] = ev.concordance_td('antolini')
results["24hr"]['antoliniadj'] = ev.concordance_td('adj_antolini')
results["24hr"]['ibs'] = ev.integrated_brier_score(time_grid)
results["24hr"]['inbll'] = ev.integrated_nbll(time_grid)
results["24hr"]['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
results["24hr"]['nbll'] = ev.nbll(time_grid).to_numpy().tolist()
results["24hr"]['iloc'] = surv.iloc().obj.to_dict()
results["24hr"]['truth'] = events_test.tolist()

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration != 12].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

durations_test, events_test = get_target(alldata_tst)
x_tst = x_mapper.transform(alldata_tst).astype('float32')
surv = model.predict_surv_df(x_tst)

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
logger.info(f'Found 12-hour concordance_td: {ev.concordance_td()}')
time_grid = np.linspace(durations_all.min(), durations_all.max(), 100)

logger.info(f'Found 12-hour integrated_brier_score: {ev.integrated_brier_score(time_grid)}')
logger.info(f'Found 12-hour integrated_nbll: {ev.integrated_nbll(time_grid)}')

results["12hr"] = dict()
results["12hr"]['antolini'] = ev.concordance_td('antolini')
results["12hr"]['antoliniadj'] = ev.concordance_td('adj_antolini')
results["12hr"]['ibs'] = ev.integrated_brier_score(time_grid)
results["12hr"]['inbll'] = ev.integrated_nbll(time_grid)
results["12hr"]['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
results["12hr"]['nbll'] = ev.nbll(time_grid).to_numpy().tolist()
results["12hr"]['iloc'] = surv.iloc().obj.to_dict()
results["12hr"]['truth'] = events_test.tolist()

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration != 6].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

durations_test, events_test = get_target(alldata_tst)
x_tst = x_mapper.transform(alldata_tst).astype('float32')
surv = model.predict_surv_df(x_tst)

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
logger.info(f'Found 6-hour concordance_td: {ev.concordance_td()}')
time_grid = np.linspace(durations_all.min(), durations_all.max(), 100)

logger.info(f'Found 6-hour integrated_brier_score: {ev.integrated_brier_score(time_grid)}')
logger.info(f'Found 6-hour integrated_nbll: {ev.integrated_nbll(time_grid)}')

results["6hr"] = dict()
results["6hr"]['antolini'] = ev.concordance_td('antolini')
results["6hr"]['antoliniadj'] = ev.concordance_td('adj_antolini')
results["6hr"]['ibs'] = ev.integrated_brier_score(time_grid)
results["6hr"]['inbll'] = ev.integrated_nbll(time_grid)
results["6hr"]['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
results["6hr"]['nbll'] = ev.nbll(time_grid).to_numpy().tolist()
results["6hr"]['iloc'] = surv.iloc().obj.to_dict()
results["6hr"]['truth'] = events_test.tolist()

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration != 3].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

durations_test, events_test = get_target(alldata_tst)
x_tst = x_mapper.transform(alldata_tst).astype('float32')
surv = model.predict_surv_df(x_tst)

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
logger.info(f'Found 3-hour concordance_td: {ev.concordance_td()}')
time_grid = np.linspace(durations_all.min(), durations_all.max(), 100)

logger.info(f'Found 3-hour integrated_brier_score: {ev.integrated_brier_score(time_grid)}')
logger.info(f'Found 3-hour integrated_nbll: {ev.integrated_nbll(time_grid)}')

results["3hr"] = dict()
results["3hr"]['antolini'] = ev.concordance_td('antolini')
results["3hr"]['antoliniadj'] = ev.concordance_td('adj_antolini')
results["3hr"]['ibs'] = ev.integrated_brier_score(time_grid)
results["3hr"]['inbll'] = ev.integrated_nbll(time_grid)
results["3hr"]['bs'] = ev.brier_score(time_grid).to_numpy().tolist()
results["3hr"]['nbll'] = ev.nbll(time_grid).to_numpy().tolist()
results["3hr"]['iloc'] = surv.iloc().obj.to_dict()
results["3hr"]['truth'] = events_test.tolist()

CLUSTERID = int(os.environ.get('CLUSTERID'))
PROCID = int(os.environ.get('PROCID'))
json_loc = here / f'{CLUSTERID}-{PROCID}-coxtime.json'

with open(json_loc, 'wt') as f:
    json.dump(results, f)

logger.info(f'Done.')
