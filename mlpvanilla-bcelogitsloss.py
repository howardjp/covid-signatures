import json
import numpy as np
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, Normalizer
from sklearn_pandas import DataFrameMapper

from spdlog import ConsoleLogger

import torchtuples as tt
from torchtuples.practical import MLPVanilla

MAXPATIENTS = 1
TRNTSTSPLIT = 0.8
TRNVALSPLIT = 0.8

logger = ConsoleLogger("mlpvanilla-bcelogitsloss.py", multithreaded=True, stdout=True, colored=True)

TORCHSEED = int(os.environ.get('TORCHSEED'))
logger.info(f'Setting torch.manual_seed to {TORCHSEED}')
torch.manual_seed(TORCHSEED)

NPSEED =int(os.environ.get('NPSEED'))
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

get_target = lambda df: (df['event'].values.astype('float32').reshape(-1, 1))
y_trn = get_target(alldata_trn)
y_val = get_target(alldata_val)
y_tst = get_target(alldata_tst)
val = tt.tuplefy(x_val, y_val)

in_features = x_trn.shape[1]
num_nodes = [64, 64]
out_features = 1
batch_norm = True
dropout = 0.1
net = MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
loss = torch.nn.BCEWithLogitsLoss()
batch_size = 256

model = tt.Model(net, loss, tt.optim.Adam)

learn_rate = 0.0001
lrfinder = model.lr_finder(x_trn, y_trn, batch_size, tolerance=2)
_ = lrfinder.plot()
learn_rate = min(lrfinder.get_best_lr(), learn_rate)
logger.info(f'Found learning rate of {lrfinder.get_best_lr()}, using learning rate of {learn_rate}')
model.optimizer.set_lr(learn_rate)

epochs = 512
callbacks = [tt.callbacks.EarlyStopping(patience=10)]
verbose = True

log = model.fit(x_trn, y_trn, batch_size, epochs, callbacks, verbose, val_data=val)

results = dict()

results["torch_seed"] = TORCHSEED
results["np_seed"] = NPSEED
results["training_history"] = json.loads(log.to_pandas().to_json())

preds = model.predict(x_tst, numpy=False).sigmoid().numpy()
cfm = confusion_matrix(y_tst, preds > 0.50)
TN, FP, FN, TP = cfm.ravel()

logger.info(f'Found 72-hour confusion matrix: {cfm}')
logger.info(f'Found 72-hour true-positives: {TP}')
logger.info(f'Found 72-hour false-positives: {FP}')
logger.info(f'Found 72-hour true-negatives: {TN}')
logger.info(f'Found 72-hour false-negatives: {FN}')

results["72hr"] = dict()
results["72hr"]['tp'] = int(TP)
results["72hr"]['fp'] = int(FP)
results["72hr"]['tn'] = int(TN)
results["72hr"]['fn'] = int(FN)

maxtimes_by_id = pd.DataFrame(alldata.groupby(['id'], sort=False)['duration'].max())
maxtimes_by_id.rename(columns={"id": "id", "duration": "maxtime"}, inplace=True)

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration < (alldata_tmp.maxtime - 23)].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

x_tst = x_mapper.transform(alldata_tst).astype('float32')
y_tst = get_target(alldata_tst)

preds = model.predict(x_tst, numpy=False).sigmoid().numpy()
cfm = confusion_matrix(y_tst, preds > 0.50)
TN, FP, FN, TP = cfm.ravel()

logger.info(f'Found 24-hour confusion matrix: {cfm}')
logger.info(f'Found 24-hour true-positives: {TP}')
logger.info(f'Found 24-hour false-positives: {FP}')
logger.info(f'Found 24-hour true-negatives: {TN}')
logger.info(f'Found 24-hour false-negatives: {FN}')

results["24hr"] = dict()
results["24hr"]['tp'] = int(TP)
results["24hr"]['fp'] = int(FP)
results["24hr"]['tn'] = int(TN)
results["24hr"]['fn'] = int(FN)

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration < (alldata_tmp.maxtime - 11)].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

x_tst = x_mapper.transform(alldata_tst).astype('float32')
y_tst = get_target(alldata_tst)

preds = model.predict(x_tst, numpy=False).sigmoid().numpy()
cfm = confusion_matrix(y_tst, preds > 0.50)
TN, FP, FN, TP = cfm.ravel()

logger.info(f'Found 12-hour confusion matrix: {cfm}')
logger.info(f'Found 12-hour true-positives: {TP}')
logger.info(f'Found 12-hour false-positives: {FP}')
logger.info(f'Found 12-hour true-negatives: {TN}')
logger.info(f'Found 12-hour false-negatives: {FN}')

results["12hr"] = dict()
results["12hr"]['tp'] = int(TP)
results["12hr"]['fp'] = int(FP)
results["12hr"]['tn'] = int(TN)
results["12hr"]['fn'] = int(FN)

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration < (alldata_tmp.maxtime - 5)].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

x_tst = x_mapper.transform(alldata_tst).astype('float32')
y_tst = get_target(alldata_tst)

preds = model.predict(x_tst, numpy=False).sigmoid().numpy()
cfm = confusion_matrix(y_tst, preds > 0.50)
TN, FP, FN, TP = cfm.ravel()

logger.info(f'Found 6-hour confusion matrix: {cfm}')
logger.info(f'Found 6-hour true-positives: {TP}')
logger.info(f'Found 6-hour false-positives: {FP}')
logger.info(f'Found 6-hour true-negatives: {TN}')
logger.info(f'Found 6-hour false-negatives: {FN}')

results["6hr"] = dict()
results["6hr"]['tp'] = int(TP)
results["6hr"]['fp'] = int(FP)
results["6hr"]['tn'] = int(TN)
results["6hr"]['fn'] = int(FN)

alldata_tmp = pd.merge(left=alldata, right=maxtimes_by_id, left_on='id', right_on='id')
alldata_tmp = alldata_tmp.drop(alldata_tmp[alldata_tmp.duration < (alldata_tmp.maxtime - 2)].index)
alldata_tst = pd.merge(left=iddata_tst.drop(columns="event"), right=alldata_tmp.drop(columns="maxtime"), left_on='id', right_on='id').drop(columns=["id"]).astype({'event': int})

x_tst = x_mapper.transform(alldata_tst).astype('float32')
y_tst = get_target(alldata_tst)

preds = model.predict(x_tst, numpy=False).sigmoid().numpy()
cfm = confusion_matrix(y_tst, preds > 0.50)
TN, FP, FN, TP = cfm.ravel()

logger.info(f'Found 3-hour confusion matrix: {cfm}')
logger.info(f'Found 3-hour true-positives: {TP}')
logger.info(f'Found 3-hour false-positives: {FP}')
logger.info(f'Found 3-hour true-negatives: {TN}')
logger.info(f'Found 3-hour false-negatives: {FN}')

results["3hr"] = dict()
results["3hr"]['tp'] = int(TP)
results["3hr"]['fp'] = int(FP)
results["3hr"]['tn'] = int(TN)
results["3hr"]['fn'] = int(FN)

CLUSTERID = int(os.environ.get('CLUSTERID'))
PROCID = int(os.environ.get('PROCID'))
json_loc = here / f'{CLUSTERID}-{PROCID}-mlpvanilla-bcelogitsloss.json'

with open(json_loc, 'wt', encoding="utf-8") as f:
    json.dump(results, f)
