# libraries
import time
import os
from  fastapi import FastAPI , Request# type: ignore
from fastapi.responses import HTMLResponse #type:ignore
from fastapi.templating import Jinja2Templates #type:ignore
from controller import ragControll

# Initialization
app = FastAPI()
RAG = ragControll()


#html template 
templates = Jinja2Templates(directory="templates")

@app.get("/",response_class=HTMLResponse)
async def main(request:Request):
 '''Root endpoint'''

 return templates.TemplateResponse("index.html",{"request":request})

# index endpoint for reprocess the documents
@app.get("/index")
async def Process():
 '''
 For reprocess the document folder
 '''
 Data = await RAG.getDataFromDocuments()

 # if documents folder haven;t any files
 if Data.get('Status Code') ==  400:
  return {
   "Status":"Not Okay",
   "Status Code":400,
   "Message":Data['Message']
  }
 # build a vocabulary file
 await RAG.buildVocabulary(Data['Message'])
# term frequency
 TF = await RAG.convertTF(Data['Message'])
# build a idf file
 await RAG.convertIDF(Data["Message"])
# final vecorization
 await RAG.TF_IDF(TF['Response'],Data['Message'])

 return {
  "Status":"Ok"
 }


@app.get("/search")
async def search(q:str):
 TF_Query = await RAG.convertTF({"query":q})
 Data = await RAG.getDataFromDocuments()

 # if documents folder haven;t any files
 if Data.get('Status Code') ==  400:
  return {
   "Status":"Not Okay",
   "Status Code":400,
   "Message":Data['Message']
  }
 resposne = await RAG.cosineSIM(TF_Query['Response'][0],Data['Message'])
 return {
  "results":resposne
 }




 
