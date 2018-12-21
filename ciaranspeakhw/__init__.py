import tarfile,sys,os
import pandas as pd
from sklearn import preprocessing, naive_bayes, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#python 2/3 compliance
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


def predict(modelfile,textDF,correct=False):
    """Loads user specified Model File and attempts to classify
    texts according to that specified model"""
    if modelfile:
        try:
            with open(modelfile,'rb') as pickle_file:
                pickle.load(pickle_file)
        except:
            sys.stderr.write(f"Error opening {modelfile}\n")
            raise
    else:
        sys.stderr.write("No model file specified.\n")
        return

    predictions = modelfile.predict(textDF)
    #if correct labels are provided, for accuracy/recall
    if correct:
        class_metrics = metrics.classification_report(predictions,correct)
        return text,predictions,class_metrics
    else:
        return text,predictions


def get_features(docDF,hold_aside=0.2):
    """This takes in a dataframe of training data, 
    splits it up for encoding purposes, and generates features 
    using TF/IDF on the ngram level"""

    if type(hold_aside) != float:
        sys.stderr.write(f"Test hold out must be entered as a float. Input: {hold_aside}\n")
    elif hold_aside > 1 or hold_aside < 0:
        sys.stderr.write(f"Test hold out must be between 0 and 1. Input: {hold_aside}\n")
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
    
    sys.stdout.write(f"{model_name} model trained using N-Gram Vectors\n")
    if not mfile_name:
        #default to currenttimestamp
        mfile_name = f"{model_name}Model{datetime.now().isoformat()[:19]}.pkl"

    with open(mfile_name,'wb') as output:
        #was going to use -1 as protocol, for highest, 
        #but this is more legible
        pickle.dump(model,output,pickle.HIGHEST_PROTOCOL)
        sys.stdout.write(f"Model written to {output}.\n")
    return model

def load_documents(doc_path):
    """Loads a directory or tar of files
    using child directory names for labels
    Returns a Pandas DataFrame
    """
    if tarfile.is_tarfile(doc_path):
        doc_path = untar(doc_path)
    
    label_list,documents = [],[]
    #get labels and docs matching that label
    for label in next(os.walk(doc_path))[1]:
        #grab the document names
        doc_names = next(os.walk(os.path.join(doc_path,label)))[2]
        #and make sure they have their path attached to them...
        doc_names = [os.path.join(doc_path,label,doc) for doc in doc_names]
        documents.extend(doc_names)
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
        sys.stderr.write(f"Unable to extract {fname}.\n")
        raise
    dir_list = []
    #Checks for child directories
    for directory in next(os.walk(extraction_path))[1]:
        #grab the one with the same assignment name as the tar
        if directory in bname:
            dir_list.append(directory)
    if len(dir_list) == 1:
        extraction_path = os.path.join(extraction_path,dir_list[0])
        sys.stdout.write(f"Tar extracted to {extraction_path}\n")
    elif len(dir_list) > 1:
        counter = 0
        while counter < 3:
            dir_counter = 1
            for directory in dir_list:
                print(f"{dir_counter}: {directory}")
            user_input = raw_input(f"Please choose an option between 1 and {len(dir_list)}\n")
            try:
                if user_input - 1 in range(len(dir_list)):
                    dir_list = [dir_list[user_input-1]]
                    continue
            except IndexError:
                sys.stdout.write(f"{user_input} is not a valid option\n")
            except TypeError:
                sys.stdout.write("Numbers only, please\n")
            counter += 1
        # ch
        extraction_path = os.path.join(extraction_path,dir_list[0])
        sys.stdout.write(f"Tar extracted to {extraction_path}\n")
        
    #not the prettiest, but I'm not certain how else to make sure that this
    #only triggers if we haven't narrowed it down.
    if len(dir_list) != 1:
        sys.stderr.write("Unclear which path to use. Exiting.\n")
        exit(1)

    return extraction_path



