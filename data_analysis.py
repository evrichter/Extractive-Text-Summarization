import os



import pandas as pd



df = pd.read_csv("data/final_train.csv")

print("training data:")
print("the number of documents:"+ str(len(df)))
df["sentence"] = df.source.apply(lambda x: len(x.split(".")))
df["words"] = df.source.apply(lambda x: len([xxx for xx in x.split(".") for xxx in xx.split(" ")]))
print("max number of sentence in a document:"+ str(df["sentence"].max()))
print("max number of words in a document:"+ str(df["words"].max()))
print("min number of sentence in a document:"+ str(df["sentence"].min()))
print("min number of words in a document:"+ str(df["words"].min()))

df = pd.read_csv("data/seen_test.csv")

print("seen test data:")
print("the number of documents:"+ str(len(df)))
df["sentence"] = df.source.apply(lambda x: len(x.split(".")))
df["words"] = df.source.apply(lambda x: len([xxx for xx in x.split(".") for xxx in xx.split(" ")]))
print("max number of sentence in a document:"+ str(df["sentence"].max()))
print("max number of words in a document:"+ str(df["words"].max()))
print("min number of sentence in a document:"+ str(df["sentence"].min()))
print("min number of words in a document:"+ str(df["words"].min()))


df = pd.read_csv("data/unseen_test.csv")
print("unseen test data:")
print("the number of documents:"+ str(len(df)))
df["sentence"] = df.source.apply(lambda x: len(x.split(".")))
df["words"] = df.source.apply(lambda x: len([xxx for xx in x.split(".") for xxx in xx.split(" ")]))
print("max number of sentence in a document:"+ str(df["sentence"].max()))
print("max number of words in a document:"+ str(df["words"].max()))
print("min number of sentence in a document:"+ str(df["sentence"].min()))
print("min number of words in a document:"+ str(df["words"].min()))

