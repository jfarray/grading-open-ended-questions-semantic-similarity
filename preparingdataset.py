import gradio as gr
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, util
import pandas as pd

def Main(Modelo, File):
  
  error = ""
  modelResult  = ""
  
  try:
    data_test = ConvertJsonToList(File.name)
    modelResult = TestModel('jfarray/Model_'+ Modelo +'_50_Epochs',data_test)
  except Exception as e:
    error = e
  
  return [error, modelResult]
    
def ConvertJsonToList(fileName):
  subject_fileDataset = load_dataset("json", data_files=fileName)

  samples = []

  for i in range (0,len(subject_fileDataset["train"])): #len(subject1)
    hashed_id = subject_fileDataset["train"][i]['hashed_id']
    mark = subject_fileDataset["train"][i]['nota']
    responseStudent = subject_fileDataset["train"][i]['respuesta']
    responseTeacher = ""
    for j in range(0,len(subject_fileDataset["train"][i]['metadata']['minipreguntas'])):
      responseTeacher = responseTeacher + subject_fileDataset["train"][i]['metadata']['minipreguntas'][j]['minirespuesta']
    
    ie = InputExample(guid= hashed_id, texts=[responseTeacher, responseStudent], label=mark)

    samples.append(ie)

  return samples
  
def TestModel(checkpoint, data):
  local_model_path = checkpoint
  model = SentenceTransformer(local_model_path)
  df = pd.DataFrame(columns=["Hashed_id", "Nota", "Similitud Semántica"])
  
  sentences1 = []
  sentences2 = []
  hashed_ids = []
  marks = []
  scores = []
  for i in range (0,len(data)): #len(data)
    sentences1.append(data[i].texts[0])
    sentences2.append(data[i].texts[1])

  #Compute embedding for both lists
  embeddings1 = model.encode(sentences1, convert_to_tensor=True)
  embeddings2 = model.encode(sentences2, convert_to_tensor=True)
  
  #Compute cosine-similarits
  cosine_scores = util.cos_sim(embeddings1, embeddings2)
  
  for i in range(len(sentences1)):
    hashed_ids.append(data[i].guid)
    marks.append(data[i].label)
    scores.append(round(cosine_scores[i][i].item(),3))

  df['Hashed_id'] = hashed_ids
  df['Nota'] = marks
  df['Similitud Semántica'] = scores
  
  return df

Modelos = gr.inputs.Dropdown(["dccuchile_bert-base-spanish-wwm-uncased"
  , "bert-base-multilingual-uncased"
  , "all-distilroberta-v1"
  , "paraphrase-multilingual-mpnet-base-v2"
  , "paraphrase-multilingual-MiniLM-L12-v2"
  , "distiluse-base-multilingual-cased-v1"])
Opciones = gr.inputs.Radio(["Comparar Textos", "Procesar Fichero"])
FileInput = gr.inputs.File(file_count="single", type="file", label="Fichero Json")
LabelOutput = gr.outputs.Label(num_top_classes=None, type="auto", label="")
DataFrameOutput = gr.outputs.Dataframe(headers=["Hashed_id", "Nota", "Similitud Semántica"]
  , max_rows=20, max_cols=None, overflow_row_behaviour="paginate", type="auto", label="Resultado")

iface = gr.Interface(fn=Main
    , inputs=[Modelos, FileInput]
    , outputs=[LabelOutput, DataFrameOutput]
    , examples=[["dccuchile_bert-base-spanish-wwm-uncased", "TestFile.json"]]
    , title = "Similitud Semántica de textos en Español de tamaño medio (200-250 palabras)"
)

iface.launch(share = True,enable_queue=True, show_error =True)