import tarfile,sys,os
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#python 2/3 compliance
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def predict(modelfile,texts,correct=False):
    """Loads user specified Model File and attempts to classify
    texts according to that specified model"""
    try:
        pickle.load(modelfile)
    except:
        sys.stderr.write(f"Error opening {modelfile}")
        raise

    #if correct labels are provided, for accuracy/recall
    if correct:
        return text,classification,accuracy
    else:
        return text,classification


def get_features(docDF,hold_aside=0.2):
    """This takes in a dataframe of training data, 
    splits it up for encoding purposes, and generates features 
    using TF/IDF on the ngram level"""

    #split defaulting to an 80/20 train/test split.
    train_x, test_x, train_y, test_y = train_test_split(docDF['text'], 
                                                docDF['label'],
                                                test_size=hold_aside)
    encoder = preprocessing.LabelEncoder()
    train_label = encoder.fit_transform(train_y)
    test_label = encoder.fit_transform(test_y)

    #setting up the vectorizer
    tfidf_vec_ngram = TfidfVectorizer(analyzer='word',
                        token_pattern=r'\w{1,}',
                        ngram_range=(2,3), 
                        max_features=5000)
    #fitting it (because verbosity)
    tfidf_vec_ngram.fit(docDF['text'])
    xtrain_tfidf_ngram =  tfidf_vec_ngram.transform(train_x)
    xtest_tfidf_ngram =  tfidf_vec_ngram.transform(test_x)

    return xtrain_tfidf_ngram, train_label, xtest_tfidf_ngram, test_label

def train_model(model_type,feature_vector_train, label,mfile_name=False):
    """Gives users options of what type of classifier to test.
    Saves model object"""
    if model_type in ["Naive Bayes","NB","Bayes","Bayesian"]:
        model = naive_bayes.MultinomialNB().fit(feature_vector_train,label)
        model_name = "NB"
    elif model_type in ["SVM","Support Vector Machine"]:
        model = svm.SVC().fit(feature_vector_train,label)
        model_name = "SVM"
    
    sys.stdout.write(f"{model_name} model trained using N-Gram Vectors")
    if not mfile_name:
        #default to currenttimestamp
        mfile_name = f"{model_name}Model{datetime.now().isoformat()[:19]}.pkl"

    with open(mfile_name,'wb') as output:
        #was going to use -1 as protocol, for highest, 
        #but this is more legible
        pickle.dump(model,output,pickle.HIGHEST_PROTOCOL)
    return model

def load_documents(doc_path,hold_aside=0.2):
    """Loads a directory or tar of files
    using child directory names for labels
    Returns a pair of Pandas DataFrames, for a  Test/Train split
    """
    if tarfile.is_tarfile(doc_path):
        doc_path = untar(doc_path)
    
    label_list,documents = [],[]
    #get labels and docs matching that label
    for label in next(os.walk(doc_path))[1]:
        #grab the document names
        documents.extend(next(os.walk(os.path.join(doc_path,label)))[2])
        #and add equivalent number of labels
        label_list.extend([label] * (len(documents) - len(label_list)))
        
    #convert to pandas dataframe
    all_docs = pd.DataFrame()
    #get the document texts
    all_docs['text'] = [open(doc).read() for doc in documents]
    all_docs['label'] = label_list

    return all_docs



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



