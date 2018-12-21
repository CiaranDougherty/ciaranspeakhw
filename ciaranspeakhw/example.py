#!/usr/bin/python3

import ciaranspeakhw,sys

filename=sys.argv[1]

if sys.argv[2]:
    hold_aside = sys.argv[2]

dataframe = ciaranspeakhw.load_documents(filename)

xtrain, label_train, xtest, label_test = get_features(dataframe)

ciaranspeakhw.train_model("NB",xtrain,label_train,"./TestModel.pkl")

ciaranspeakhw.predict("./TestModel.pkl",xtest,label_test)