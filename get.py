#!/usr/bin/env python3

import os
import pathlib
import urllib.request
import zipfile

here = pathlib.Path(__file__).resolve().parent

base_base_loc = here / 'data'
base_loc = base_base_loc / 'sepsis'
loc_Azip = base_loc / 'training_setA.zip'
loc_Bzip = base_loc / 'training_setB.zip'

def download():
    if not os.path.exists(loc_Azip):
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setA.zip',
                                   str(loc_Azip))
        urllib.request.urlretrieve('https://archive.physionet.org/users/shared/challenge-2019/training_setB.zip',
                                   str(loc_Bzip))

        with zipfile.ZipFile(loc_Azip, 'r') as f:
            f.extractall(str(base_loc))
        with zipfile.ZipFile(loc_Bzip, 'r') as f:
            f.extractall(str(base_loc))
        for folder in ('training', 'training_setB'):
            for filename in os.listdir(base_loc / folder):
                if os.path.exists(base_loc / filename):
                    raise RuntimeError
                os.rename(base_loc / folder / filename, base_loc / filename)


if not loc_Azip.is_file() or not loc_Bzip.is_file():
    download()

with zipfile.ZipFile(loc_Azip, 'r') as zip_ref:
    zip_ref.extractall(base_loc)
with zipfile.ZipFile(loc_Bzip, 'r') as zip_ref:
    zip_ref.extractall(base_loc)


