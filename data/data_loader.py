"""
Utils for data loading and transforming
"""
from pandas import read_csv
from zipfile import ZipFile
from wget import download
import numpy as np
import os


def get_data(language="eng", url="https://github.com/Gci04/AML-DS-2021/raw/main/data/nameClassification.zip"):
    """
    Download, if needed, nameClassification.zip as data folder and return
    the content of .csv files

    :param language: postfix to .csv files, determining the language of names
    :param url: url of .zip file
    :return: pandas DataFrames, train and test sets of corresponding language
    """
    data_path = os.path.join(os.pardir, "data")
    if "data" not in os.listdir(path=data_path):
        download(url)
        with ZipFile("nameClassification.zip", 'r') as zipObj:
            zipObj.extractall()
        os.remove("nameClassification.zip")
    train = read_csv(os.path.join(data_path, "data", f"train_{language}.csv"), encoding="utf-8")
    test = read_csv(os.path.join(data_path, "data",  f"test_{language}.csv"), encoding="utf-8")
    return train, test


def normalize(frame):
    """
    Convert string representation of data to numpy arrays, where:

    x - names, each name is 15x1 integer vector, where x[i] - unicode of i-th character or padding zero;

    y - genders in binary (0 or 1) form.

    :param frame: pandas DataFrame with columns Name and Gender
    :return: x, y
    """
    x = []
    y = []
    for ind, row in frame.iterrows():
        new_word = [ord(letter) for letter in row["Name"]]
        x.append(np.pad(new_word, pad_width=(0, 15 - len(new_word))))
        y.append(row["Gender"])
    # To categorical data
    _, y = np.unique(y, return_inverse=True)
    y = [np.array([c]) for c in y]
    return np.array(x), np.array(y)
