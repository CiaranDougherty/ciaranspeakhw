import pandas as pd
from sklearn import model_selection


def predict(trained_model="foo",text="bar"):
    """DocString for Predict function"""
    return trained_model,text

def train_model(training="sna",paramaters="fu"):
    """DocString for model training"""

    #train model
    trained_model = training + paramaters
    return trained_model,results

def load_documents():
    """Load a directory or tar of files 
    """
    #convert to dataframe

    #set aside 20% as test set
    train_set, test_set = model_selection.train_test_split(df, test_size=0.2)


import tarfile,sys

#untar as necessary
def untar(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname)
        tar.extractall()
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])



