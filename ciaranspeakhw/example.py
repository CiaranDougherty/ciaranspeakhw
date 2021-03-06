#!/usr/bin/python3

import ciaranspeakhw,sys

filename=sys.argv[1]

if len(sys.argv) > 2:
    hold_aside = sys.argv[2]

dataframe = ciaranspeakhw.load_documents(filename)
print("Dataframe loaded")

xtrain, label_train, xtest, label_test = ciaranspeakhw.get_features(dataframe)

ciaranspeakhw.train_model("NB",xtrain,label_train,"./TestModel.pkl")

print(ciaranspeakhw.predict("./TestModel.pkl",xtest,label_test))