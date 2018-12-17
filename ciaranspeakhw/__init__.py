import tarfile,sys,os
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

def load_documents(doc_path,hold_aside=0.2):
    """Loads a directory or tar of files
    using child directory names for labels
    Returns a pair of Pandas DataForms, for a  Test/Train split
    """
    if tarfile.is_tarfile(doc_path):
        doc_path = untar(doc_path)
    
    #convert to pandas dataframe

    #set aside 20% as test set
    train_set, test_set = model_selection.train_test_split(df, hold_aside=0.2)
    return train_set, test_set



def untar(fname):
    """Untars provided files into an 'extracted' folder"""
    filepath = os.path.dirname(fname)
    bname = os.path.basename(fname)
    extraction_path=os.path.join(filepath,'Extracted')
    os.mkdir(extraction_path)
    try:
        tar = tarfile.open(fname)
        tar.extractall(extraction_path)
        tar.close()
    except TarError as e:
        sys.stderr.write(f"Unable to extract {fname}.")
        raise
    dir_list = []
    #Checks for child directories
    for directory in next(os.walk(filepath))[1]:
        #grab the one with the same assignment name as the tar
        if directory in bname:
            dir_list.append(directory)
    if len(dir_list) == 1:
        extraction_path = os.path.join(extraction_path,dir_list[0])
        sys.stdout.write(f"Tar extracted to {extraction_path}")
    else:
        #I should properly make this a "request user input" scenario
        # with [os.path.join(extraction_path,d) for d in dir_list] or similar
        # but between the power outage, and other cases of life,
        # I'm running short on time for this...
        sys.stderr.write(f"Unclear which path to use")
        exit(1)

    return extraction_path



