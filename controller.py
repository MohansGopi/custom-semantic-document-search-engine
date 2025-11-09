# libraries
import os
import json
import string
import time
import math
import numpy as np
from dotenv import load_dotenv

# Initialization
load_dotenv()



DOCUMENT_PATH = os.getenv("PATH_DOCUMENTS")

class ragControll:
 '''*ragControll* have set of methods like getDataFromDocuments,  etc..'''

 def __init__(self):
    self.VectorName = ''
    self.IDFName = ''
    self.vocabName = ''
 

 async def getDataFromDocuments(self):
  '''getDataFromDocuments is to retrive the data from each document and store it to the list'''

  files = os.listdir(DOCUMENT_PATH) # retrive all the files inside the documents path

  if not files:
   return {
    "Status Code" : 400,
    "Message": "Please upload any files"
   }

  TextAndFilenameFromDocuments = {} # for store the filename and the text inside the file anme

  for fileName in files:
   # retrive and store
   with open(f"{DOCUMENT_PATH}/{fileName}",'r') as Text:
    topic = "_".join(fileName.split('_')[2:]) if "doc_" in fileName else fileName  # spliting the fileName for removing the doc_01_
    topic = topic.replace('.txt', '')
    TextAndFilenameFromDocuments[topic] = Text.read()
  
  return {
   "Status Code":200,
   "Message":TextAndFilenameFromDocuments
   
  }
 

 async def buildVocabulary(self,Data:dict):
  '''
  buildVocabulary is to create a vocab file for the model and the vocab file is used to vectorize the model
  '''

  vocab = [] # for store the vocabulary

  # stopword removal
  stopwords_path = os.path.join(os.getenv("MODEL_FOLDER"), "stopwrods.txt")
  if not os.path.exists(stopwords_path):
        return {"Status Code": 400, "Response": "stopwords not found in model folder"}
  
  with open(f"{os.getenv('MODEL_FOLDER')}/stopwrods.txt",'r') as stopwords:
   stWords = [word.strip() for word in stopwords.readlines()]
  

  for value in Data.values():
   value = (value.translate(str.maketrans('', '', string.punctuation))).replace("\n"," ") # Remove punctuation and remove /n to get clean text
   
   for word in (word.lower() for word in value.split(" ") if word and word.lower() not in vocab):
     if word not in stWords:
      vocab.append(word)

  # adding the vocab file to the model
  self.vocabName = "vocab"+str(time.time())+".txt"

  with open(f"{os.getenv("MODEL_FOLDER")}/{self.vocabName}",'w') as vocabFile:
   for word in sorted(vocab):
    vocabFile.write(word+"\n")
  
  return {
   "Status Code":200,
   "Message":"success"
  }
 
 async def convertTF(self,docs:dict):
   '''
   convertTF is convert the text into term frequency(TF)
   '''
   # get the vocab file data

   if self.vocabName == '' or self.vocabName not in os.listdir(os.getenv("MODEL_FOLDER")):
    return {
     "Status Code":400,
     "Response":"Need to build vocabulory file"
    }
   

   with open(f"{os.getenv("MODEL_FOLDER")}/{self.vocabName}",'r') as vocabData:
    vocab = vocabData.readlines()
  
  # stop word removal for query
   if docs.get("query"):
    stopwords_path = os.path.join(os.getenv("MODEL_FOLDER"), "stopwrods.txt")
    with open(stopwords_path, "r") as stopwords_file:
      stopwords_list = set(stopwords_file.read().splitlines())

   
   # get the text data from each doc in documents and make it tf
   TF = []
   for value in docs.values():
    text = value.translate(str.maketrans('', '', string.punctuation)).replace("\n", " ")

    words = text.split()
    if "query" in docs.keys():
        words = [w.lower() for w in words if w and w.lower() not in stopwords_list]
    totalNo_OfWordsInText = len(words)

    docTF = [] # 

    for word in vocab:
     if not word:
      continue
     count = words.count(word.strip())
     docTF.append(count/totalNo_OfWordsInText)
    TF.append(docTF)
   
   return {
    "Status Code":200,
    "Response":TF
   }

 async def convertIDF(self,docs:dict):
  '''
  convertIDF is to covert the vocab text in the Inverse document term
  '''
  # get the words in vocab file
  with open(f"{os.getenv("MODEL_FOLDER")}/{self.vocabName}",'r') as vocabData:
   vocabText = [word.strip() for word in vocabData.readlines()]  
  
  IDF_ = []

  for word in vocabText:
    count = 0
    # get doc from documents and clean it
    for value in docs.values():

      text = (value.translate(str.maketrans('', '', string.punctuation))).replace("\n"," ") # make the text clean

      if word in text.split(" "):
       count+=1
    totalNo_Documebt = len(docs.values())

    # IDF(word)

    IDFword = math.log((1+totalNo_Documebt)/(1+count)+1)

    IDF_.append(IDFword) # IDF for each word in vocab
  
  self.IDFName = "IDF"+str(time.time())+".txt"

  with open(f"{os.getenv("MODEL_FOLDER")}/{self.IDFName}",'w') as IDFfile:
    for IDF in IDF_:
      IDFfile.write(str(IDF)+"\n")

  return {
   "Status Code": 200,
   "Response":IDF_ 
  }

 async def TF_IDF(self,TF:list,Docs:dict):
    '''
    TF_IDF is used to vectorize the text in docs
    '''

    # Get the Idf list

    with open(f"{os.getenv("MODEL_FOLDER")}/{self.IDFName}",'r') as IDFfile:
      IDF = [float(freq) for freq in IDFfile.readlines()]
    
    VectorizedDoc = dict()

    # multiply the tf and idf

    for TFreq,key in zip(TF,Docs.keys()):
      TF_IDF = np.multiply(TFreq,IDF)
    
      VectorizedDoc[key] = TF_IDF.tolist()  # add the vector to the approriate keys in docs

    # add the vectorized json to the model file
    self.VectorName = "VectorizedDocs"+str(time.time())+".txt"

    with open(f"{os.getenv("MODEL_FOLDER")}/{self.VectorName}",'w') as Vdocs:
     json.dump(VectorizedDoc,Vdocs,indent=4)
    
    return {
     "Status Code":200,
     "Status":"OK"
    }
 
 async def cosineSIM(self,Q_Vector:list,docs:dict):
  '''
  cosineSIM -> Cosine Similarity is to compare the query vector with each document vector

  '''
  # if docs are not vectorized
  if self.VectorName == '' or self.VectorName not in os.listdir(f"{os.getenv("MODEL_FOLDER")}"):
    return {
     "Status Code":400,
     "Response":"Need to upload docs"
    }
  
  
  # get the vectorized data from the vector database
  with open(f"{os.getenv('MODEL_FOLDER')}/{self.VectorName}",'r') as VectorData:
   vectorDoc = json.loads(VectorData.read())
  
  
  # store the cosine sim score
  score = []

  for key in vectorDoc.keys():
  #  calculate cosine similarities btwn two vectors 
   cosine = np.dot(Q_Vector,vectorDoc[key])/(np.linalg.norm(Q_Vector)*np.linalg.norm(vectorDoc[key]))

   score.append([key,cosine])
  # sort the score max ---> min
  scorer = (sorted(score, key=lambda x: x[1], reverse=True))[:3]
  # filter the top 3% from the sorted score
  result = []
  for key in docs.keys():
    for fName,score in scorer:
      if fName in key:
        result.append({
          "document":fName+".txt",
          "score":float(score),
          "Snippet":"".join(docs[key].split(".")[:2])
                        })
  result = sorted(result, key=lambda x: x['score'], reverse=True)
  return result


  













  





    
     
    
  







  





  




   


