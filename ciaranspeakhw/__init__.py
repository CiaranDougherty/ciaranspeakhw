import pandas as pd


def testing():
    """Stupid toy to ensure I understand this
    whole 'package' thing correctly"""
    return ("hey, it works")

def predict(trained_model="foo",text="bar"):
    """DocString for Predict function"""
    return trained_model,text

def train_model(training="sna",paramaters="fu"):
    """DocString for model training"""
    trained_model = training + paramaters
    return trained_model

import tarfile,sys

#untar as necessary
def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print "Extracted in Current Directory"
    else:
        print "Not a tar.gz file: '%s '" % sys.argv[0]



