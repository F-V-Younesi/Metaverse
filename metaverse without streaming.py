
"""#STT
##STT: SYSTRAN/faster-whisper (wav is faster) have same quality as whisper
"""

!pip install faster-whisper
!pip install ctranslate2==4.4.0

from faster_whisper import WhisperModel
import time

model_size = "large-v3"
#For GPU with FP16:
model = WhisperModel(model_size, device="cuda", compute_type="float16")

s=time.time()
segments, info = model.transcribe("/content/صدا ۰۳۸.wav", beam_size=5)
text=''
for segment in segments:
  text=text+(segment.text)
e=time.time()
e-s
text

s=time.time()
t=''
for segment in segments:
    t=t+segment.text
e=time.time()
e-s



"""##SA STT """
from transformers import AutoProcessor, AutoModelForCTC
import torch

processor = AutoProcessor.from_pretrained("SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1")
model = AutoModelForCTC.from_pretrained("SeyedAli/Persian-Speech-Transcription-Wav2Vec2-V1")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
segments= model.to(device)('/content/صدا ۰۳۹.wav')



"""##original whisper"""
!pip install whisper
!pip install openai-whisper

import whisper
modell = whisper.load_model("large")

start_time = time.time()
resultl = modell.transcribe("/content/صدا ۰۳۷.wav")
end_time = time.time()
end_time-start_time

resultl['text']



"""##speech recognition package"""
! pip install SpeechRecognition

import speech_recognition as sr
sr.__version__

s=time.time()
r = sr.Recognizer()
harvard = sr.AudioFile('/content/en-wav.wav')
with harvard as source:
    audio = r.record(source)
r.recognize_google(audio)
e=time.time()
e-s



"""
#Spell Correction"""

!pip install parsivar
from parsivar import SpellCheck
!mkdir '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell'
!cp '/content/drive/MyDrive/colab_env/lib/python3.10/site-packages/parsivar/resource/spell/onegram.pckl' '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell'
!cp '/content/drive/MyDrive/colab_env/lib/python3.10/site-packages/parsivar/resource/spell/mybigram_lm.pckl' '/usr/local/lib/python3.10/dist-packages/parsivar/resource/spell'
checker=SpellCheck()

s=time.time()
corrected_text=checker.spell_corrector('آفرین امتیاز تو نشون میده که موتور و اسکیل خوبی داری. به تمنیت ادامه بده تا محاوتت تضعیف نشه')
# corrected_text
e=time.time()
e-s
corrected_text



"""#Emotion Detection model

##Bektash: from text(faster than m3hrdadfi and more accurate than sa) ParsBert is faster
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# tokenizer = AutoTokenizer.from_pretrained("Baktashans/Finetuned_xlm_roberta_large_ArmanEmo")
# model_text = AutoModelForSequenceClassification.from_pretrained("Baktashans/Finetuned_xlm_roberta_large_ArmanEmo")
# tokenizer = AutoTokenizer.from_pretrained("Baktashans/Finetuned_ParsBert_ArmanEmo")
# model_text = AutoModelForSequenceClassification.from_pretrained("Baktashans/Finetuned_ParsBert_ArmanEmo")
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/ElmOSanat_qwe/4-meta/emotion_from_text")
model_text = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/ElmOSanat_qwe/4-meta/emotion_from_text")

def predict_labels(model, tokenizer, texts):

    labels = ['ANGRY', 'FEAR', 'HAPPY', 'HATE', 'OTHER', 'SAD', 'SURPRISE']
    model.eval()
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)

    #model's predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    #Get the predicted labels
    _, preds = torch.max(outputs.logits, dim=-1)

    return [labels[item] for item in preds.tolist()]

s=time.time()

predicted_labels = predict_labels(model_text, tokenizer, texts)
# print(predicted_labels)
e=time.time()
e-s
predicted_labels



"""##aliyzd95.speech-emotion-svm
"""
!pip install opensmile
!pip install scikit-optimize

import os
import random
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsOneClassifier
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
import warnings
from opensmile_preprocessing import opensmile_Functionals, emo_labels

warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = str(seed_value)
seed_value = 42
random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cnf_matrix


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=emo_labels, normalize=True, title='SVM + eGeMAPS')
    plt.show()


def svm(X, y):
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_value)
    cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed_value)
    model = SVC()
    ovo = OneVsOneClassifier(model)
    space = dict()
    space['estimator__C'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    space['estimator__gamma'] = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
    search = BayesSearchCV(ovo, space, scoring='recall_macro', cv=cv_inner, n_jobs=-1, verbose=0)
    pipeline = make_pipeline(StandardScaler(), search)
    scores = cross_validate(pipeline, X, y, scoring=['recall_macro', 'accuracy'], cv=cv_outer, n_jobs=-1, verbose=2)
    print('____________________ Support Vector Machine ____________________')
    print(f"Weighted Accuracy: {np.mean(scores['test_accuracy'] * 100)}")
    print(f"Unweighted Accuracy: {np.mean(scores['test_recall_macro']) * 100}")


X, y = opensmile_Functionals()

N_SAMPLES = X.shape[0]

perm = np.random.permutation(N_SAMPLES)
X = X[perm]
y = y[perm]

if __name__ == '__main__':
    svm(X, y)

# Load model directly
# from transformers import AutoProcessor, Wav2Vec2ForSpeechClassification
config = AutoConfig.from_pretrained('aliyzd95/wav2vec2-xlsr-shemo-speech-emotion-recognition')
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('aliyzd95/wav2vec2-xlsr-shemo-speech-emotion-recognition')
sampling_rate = feature_extractor.sampling_rate
proce_Spssor = AutoProcessor.from_pretrained("aliyzd95/wav2vec2-xlsr-shemo-speech-emotion-recognition")
modeleechClassification = Wav2Vec2ForSpeechClassification.from_pretrained("aliyzd95/wav2vec2-xlsr-shemo-speech-emotion-recognition")



"""##m3hrdadfi: emotion model from speech"""

# requirement packages
# !pip install git+https://github.com/huggingface/datasets.git
!pip install git+https://github.com/huggingface/transformers.git
!pip install torchaudio
!pip install librosa
!git clone https://github.com/m3hrdadfi/soxan.git

import torch.nn.functional as F
import torchaudio
import librosa
import IPython.display as ipd
import numpy as np
import pandas as pd
from transformers import AutoConfig, Wav2Vec2FeatureExtractor

!cp -r '/content/soxan/src' '/content/'

from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/hubert-base-persian-speech-emotion-recognition"
# model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-persian-speech-emotion-recognition"

config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model_SpeechClassification = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)
# model =Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model_SpeechClassification(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Label": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
    return outputs

start_time = time.time()
path = "/content/صدا ۰۳۹.wav"
outputs = predict(path, sampling_rate)
end_time = time.time()
end_time-start_time
outputs

max=0
for i in range(0,len(outputs)):
  s=float(outputs[i]['Score'].replace('%',''))
  if  s> max:
    max=s
    max_label=outputs[i]['Label']
max_label


"""##SA: emotion model from text"""

# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("SeyedAli/Persian-Text-Emotion-Bert-V1")
model = AutoModelForSequenceClassification.from_pretrained("SeyedAli/Persian-Text-Emotion-Bert-V1")

s=time.time()
# texts = ['به تمنیت ادامه بده تا محاوتت تضعیف نشه. آفرین امتیاز تو نشون میده که موتور و اسکیل خوبی داری']
texts=['ای بابا. هرچی بیشتر بازی رو انجام میدم هیچ تاثیری روی امتیازم نمیبینم. باید چه کار کنم؟']
predicted_labels = predict_labels(model, tokenizer, texts)
# print(predicted_labels)
e=time.time()
print(e-s)
predicted_labels



"""#Decision making
##Maral (Mistral) with hf (very slow)
"""

from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="MaralGPT/Maral-7B-alpha-1", filename="model-00002-of-00008.safetensors",local_dir ='/content/maral')
pip install accelerate
pip install -U bitsandbytes
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# import torch

# model_name_or_id = "MaralGPT/Maral-7B-alpha-1"
# model = AutoModelForCausalLM.from_pretrained(model_name_or_id, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)

prompt = ""
prompt = f"### Human:{prompt}\n### Assistant:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.5,
    max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



"""##ava-LLama3 v2 with hf"""
pip install -U bitsandbytes

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

model_name_or_id = "MehdiHosseiniMoghadam/AVA-Llama-3-V2"
l_model = AutoModelForCausalLM.from_pretrained(model_name_or_id, torch_dtype=torch.bfloat16, device_map="auto",load_in_8bit=True)
# model = AutoModelForCausalLM.from_pretrained(model_name_or_id, load_in_8bit=True, torch_dtype=torch.bfloat16, device_map="auto")
l_tokenizer = AutoTokenizer.from_pretrained(model_name_or_id)

prompt = ""
prompt = f"### Human:{prompt}\n### Assistant:"
inputs = l_tokenizer(prompt, return_tensors="pt").to("cuda")
generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.5,
    max_new_tokens=300,
    pad_token_id=l_tokenizer.eos_token_id)
outputs = l_model.generate(**inputs, generation_config=generation_config)
print(l_tokenizer.decode(outputs[0], skip_special_tokens=True))




"""##GPT 2 with hf"""
# from transformers import pipeline
# generator = pipeline("text-generation")
generator('پایتخت ایران کجاست؟',num_return_sequences = 1, max_length = 100)
prompt = ""
prompt = f"### Human:{prompt}\n### Assistant:"
generator(prompt,num_return_sequences = 1, max_length = 100)



"""## text generation Qwen2.5"""
# Use a pipeline as a high-level helper
# from transformers import pipeline

messages = [
    {"role": "user", "content": "where is Paris?"},
]
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B", device="cuda")
pipe(messages)




"""##hf gemma farsi(very slow)"""
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

# model_id = "alibidaran/Gemma2_Farsi"
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id,device_map={"":0})
prompt = " پایتخت ایران کجاست؟"
text = f"<s> {prompt} ###Answer: "
inputs=tokenizer(text,return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs=model.generate(**inputs,max_new_tokens=400,do_sample=True,top_p=0.99,top_k=10,temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# import t5.data.mixtures
# question_1 = "Where is the Google headquarters located?"
# # question_2 = "What is the most populous country in the world?" #@param {type:"string"}
# # question_3 = "Who are the 4 members of The Beatles?" #@param {type:"string"}
# # question_4 = "How many teeth do humans have?" #@param {type:"string"}
# questions = [question_1]

now = time.time()
predict_inputs_path = os.path.join(MODEL_DIR, "predict_inputs_%d.txt" % now)
predict_outputs_path = os.path.join(MODEL_DIR, "predict_outputs_%d.txt" % now)
# Manually apply preprocessing by prepending "triviaqa question:".
with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
  for q in questions:
    f.write("trivia question: %s\n" % q.lower())

with tf_verbosity_level('ERROR'):
  model.batch_size = 8  # Min size for small model on v2-8 with parallelism 1.
  model.predict(
      input_file=predict_inputs_path,
      output_file=predict_outputs_path,
      temperature=0,
  )

prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
print("\nPredictions using checkpoint %s:\n" % prediction_files[-1].split("-")[-1])
with tf.io.gfile.GFile(prediction_files[-1]) as f:
  for q, a in zip(questions, f):
    if q:
      print("Q: " + q)
      print("A: " + a)
      print()


"""##langchain t5"""
# !pip install huggingface_hub
# !pip install transoformers
# !pip install accelerate
# !pip install bitsandbytes
# !pip install langchain
# !pip install langchain-community
# !pip install langchain-core
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
# from langchain import PromptTemplate, LLMChain, HuggingFaceHub
# import os
!ssh-keygen -t ed25519 -C ""
os.environ['HUGGINGFACEHUB_API_TOKEN']=""

# prompt = PromptTemplate(input_varaibles=["state"],template="Can you tell me the capital of {state} from India Country")
chain = LLMChain(llm=HuggingFaceHub(repo_id='google/flan-t5-large',model_kwargs={'temperature':0}),prompt=prompt)
chain.run ("Telangana")
template = """where is iran?
Response:"""
prompt = PromptTemplate.from_template(template)
repo_id = "google/flan-t5-xxl"
llm = HuggingFaceHub(
 repo_id=repo_id,
 model_kwargs={"temperature": 0.1, "max_length": 64}
)
# memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True)
# To start a conversation
# while True:
#     query = input("User-query: ")
#     # Enter 'exit' to stop
#     if query.lower() == "exit":
#         break
#     print(conversation({"question": query})['text'])

conversation({"question": 'where is iran'})['text']




"""## langchain cohere (need api key)"""
pip install langchain-cohere
import os
os.environ('CO_API_KEY')==''
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

llm = ChatCohere()
messages = [HumanMessage(content="Hello, can you introduce yourself?")]
print(llm.invoke(messages))


"""##langchain openai(need api key)"""
pip install langchain langchain_community openai
import os
os.environ['OPENAI_API_KEY']=''
from langchain.llms import OpenAI

llm=OpenAI(temperature=0.8)
r=llm.predict('پایتخت ایران کجاست؟')
r


"""##langchain ollama"""
from langchain.llms import Ollama
from langchain.callbacks.manager import  CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="llama2",callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
llm('پایتخت ایران کجاست؟')


"""##gpe-neo (irrelevent answer)"""
# from transformers import pipeline
pipe=pipeline('text-generation',model='EleutherAI/gpt-neo-1.3B',device='cuda')
pipe('where is paris?')[0]['generated_text']


"""##QAbert"""
from huggingface_hub import snapshot_download
snapshot_download(repo_id="marzinouri/parsbert-finetuned-persianQA",local_dir="content")
from transformers import pipeline

# qa_pipeline = pipeline("question-answering", model="marzinouri/parsbert-finetuned-persianQA")
qa_pipeline = pipeline("question-answering", model="/content/drive/MyDrive/ElmOSanat_qwe/4-meta/parsbert-finetuned-persianQA",device='cuda')
context = "پایتخت ایران تهران است."
question = "شرکت فولاد مبارکه در کجا واقع شده است؟"
answer = qa_pipeline(question=question, context=context)
print( answer["answer"])
s=time.time()
question = "من امتیاز 73 گرفنم. امتیاز من خوب است؟"
answer = qa_pipeline(question=question, context=context)
e=time.time()
e-s


"""## qa ayyobi xlm-roberta"""
!pip install transformers sentencepiece
from transformers import pipeline

model_name = "SajjadAyoubi/xlm-roberta-large-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
text = "پایتخت فرانسه پاریس است."
questions = ["شرکت فولاد مبارکه در کجا واقع شده است؟"]

for question in questions:
    print(qa_pipeline({"context": text, "question": question}))



"""## qa ayyobi bert"""
from transformers import pipeline

model_name = "SajjadAyoubi/bert-base-fa-qa"
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
text = "پایتخت فرانسه پاریس است."
questions = ["شرکت فولاد مبارکه در کجا واقع شده است؟"]
for question in questions:
    print(qa_pipeline({"context": text, "question": question}))


"""##persian nlu mt5
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
model_size = "large"
model_name = f"persiannlp/mt5-{model_size}-parsinlu-squad-reading-comprehension"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)
def run_model(paragraph, question, **generator_args):
    input_ids = tokenizer.encode(question + "\n" + paragraph, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    output = tokenizer.batch_decode(res, skip_special_tokens=True)
    print(output)
    return output
run_model("یک شی را دارای تقارن می‌نامیم زمانی که ان شی را بتوان به دو یا چند قسمت تقسیم کرد که آن‌ها قسمتی از یک طرح سازمان یافته باشند یعنی بر روی شکل تنها جابجایی و چرخش و بازتاب و تجانس انجام شود و در اصل شکل تغییری به وجود نیایید آنگاه ان را تقارن می‌نامیم مرکز تقارن:اگر در یک شکل نقطه‌ای مانندA وجود داشته باشد که هر نقطهٔ روی شکل (محیط) نسبت به نقطه یAمتقارن یک نقطهٔ دیگر شکل (محیط) باشد، نقطهٔ Aمرکز تقارن است. یعنی هر نقطه روی شکل باید متقارنی داشته باشد شکل‌های که منتظم هستند و زوج ضلع دارند دارای مرکز تقارند ولی شکل‌های فرد ضلعی منتظم مرکز تقارن ندارند. متوازی‌الأضلاع و دایره یک مرکز تقارن دارند ممکن است یک شکل خط تقارن نداشته باشد ولی مرکز تقارن داشته باشد. (منبع:س. گ)","اشکالی که یک مرکز تقارن دارند")

run_model("یک شی را دارای تقارن می‌نامیم زمانی که ان شی را بتوان به دو یا چند قسمت تقسیم کرد که آن‌ها قسمتی از یک طرح سازمان یافته باشند یعنی بر روی شکل تنها جابجایی و چرخش و بازتاب و تجانس انجام شود و در اصل شکل تغییری به وجود نیایید آنگاه ان را تقارن می‌نامیم مرکز تقارن:اگر در یک شکل نقطه‌ای مانندA وجود داشته باشد که هر نقطهٔ روی شکل (محیط) نسبت به نقطه یAمتقارن یک نقطهٔ دیگر شکل (محیط) باشد، نقطهٔ Aمرکز تقارن است. یعنی هر نقطه روی شکل باید متقارنی داشته باشد شکل‌های که منتظم هستند و زوج ضلع دارند دارای مرکز تقارند ولی شکل‌های فرد ضلعی منتظم مرکز تقارن ندارند. متوازی‌الأضلاع و دایره یک مرکز تقارن دارند ممکن است یک شکل خط تقارن نداشته باشد ولی مرکز تقارن داشته باشد. (منبع:س. گ)","شرکت فولاد مبارکه در کجا واقع شده است")



"""##mbert qa"""
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = '''پایتخت ایران کجاست؟'''
paragraph = ''' پایتخت ایران تهران  است '''
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph)
inputs = encoding['input_ids']  #Token embeddings
sentence_embedding = encoding['token_type_ids']  #Segment embeddings
tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
output = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
start_index = torch.argmax(output.start_logits)
end_index = torch.argmax(output.end_logits)
answer = ' '.join(tokens[start_index:end_index+1])
answer


"""##mbert"""
from transformers import BertTokenizer, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
# device = torch.device("cpu")
device='cuda'
input_text = "پایتخت ایران کجاست؟"
tokenized_text = tokenizer.tokenize(input_text)

# Loop through each token in the input text
for i in range(len(tokenized_text)):
    # If the token is a mask, replace it with the predicted token
    if tokenized_text[i] == '[MASK]':
        # Create a copy of the tokenized text and replace the mask with a placeholder token
        masked_tokenized_text = tokenized_text.copy()
        masked_tokenized_text[i] = '[PREDICT]'

        # Convert the masked tokenized text to a tensor of token ids
        masked_index = i
        masked_index_tensor = torch.tensor([masked_index])
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Move the tokens tensor to the CPU
        tokens_tensor = tokens_tensor.to(device)

        # Generate predictions for the masked token using the model
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0][0, masked_index_tensor].topk(5)

        # Convert the predicted token ids to tokens
        predicted_token_ids = predictions.indices.tolist()
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids)

        # Replace the placeholder token with the predicted token
        tokenized_text[i] = predicted_tokens[0]

# Convert the tokenized text to a string
predicted_text = tokenizer.convert_tokens_to_string(tokenized_text)
# Print the predicted text
print(predicted_text)


"""##bert-large-cased"""
from transformers import AutoModelForCausalLM, AutoTokenizer

# load the model and tokenizer
model_name = "bert-large-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "پایتخت ایران کجاست؟ "
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=50, do_sample=True)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)

"""##bert-base-multilingual-cased"""
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
text = "پایتخت ایران کجاست؟"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
output

"""##bert-base-uncased"""
pip install langchain
from transformers import BertModel, BertTokenizer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "where is paris?"
tokens = tokenizer(text, return_tensors='pt')
prompt = PromptTemplate(template="Classify the following text: {text}")

class BERTClassifier(LLMChain):
    def _call(self, inputs):
        tokens = tokenizer(inputs['text'], return_tensors='pt')
        outputs = model(**tokens)
        return outputs

tokens = tokenizer("where is paris?", return_tensors='pt')
outputs = model(**tokens)
output_text=tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(output)

bc=BERTClassifier()
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)
answer = ' '.join(tokens[start_index:end_index+1])
answer



"""#TTS . best for persian:microsoft: best at time
"""

!pip install edge-tts

answer="در این بازی مکعب ها به صورت شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن، خونه هایی که یادت مونده رنگی بودن رو انتخاب کن"
s=time.time()
# !edge-tts --voice fa-IR-DilaraNeural --text "در این بازی مکعب ها به صورت شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن، خونه هایی که یادت مونده رنگی بودن رو انتخاب کن" --write-media microsoft_per_tts.mp3 --write-subtitles hello.vtt
!edge-tts --voice fa-IR-FaridNeural --text "در این بازی مکعب ها به صورت شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن، خونه هایی که یادت مونده رنگی بودن رو انتخاب کن" --write-media microsoft_per_tts.mp3 --write-subtitles hello.vtt
e=time.time()
e-s


"""##persian piper (add some words!!for amir model) ganji is good  fast and high quality."""

from huggingface_hub import snapshot_download
snapshot_download(repo_id="gyroing/Persian-Piper-Model-gyro", local_dir="content")
# hf_hub_download(repo_id="SadeghK/persian-text-to-speech", filename="farsi/amir/epoch=5261-step=2455712.onnx.json",local_dir="content")
# https://huggingface.co/datasets/SadeghK/datacula-pertts-amir/resolve/main/pertts-speech-database-rokh-ljspeech.zip
# https://huggingface.co/datasets/SadeghK/datacula-pertts-amir/resolve/main/pertts-speech-database-rokh-ljspeech.zip.
# hf_hub_download(repo_id="SadeghK/datacula-pertts-amir", filename="pertts-speech-database-rokh-ljspeech.zip",local_dir="content",repo_type='dataset')
!pip install piper-tts

import wave
from piper import PiperVoice

model_path = "/content/content/fa_IR-gyro-medium.onnx"
config_path = "/content/content/fa_IR-gyro-medium.onnx.json"

# Load voice
voice = PiperVoice.load(model_path, config_path=config_path)
synthesize_args = {}

s=time.time()
# Read entire input
# text = "در این بازی مکعب ها به صورتِ شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن،خونه هایی که یادت مونده که رنگی بودن رو انتخاب کن"
text="در این بازی مکعب ها به صورتِ شَبَکِه ای کنار هم قرار میگیرند. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدند، خونه هایی که یادِت مونده که رنگی بودن انتخاب کنْ"
with wave.open("piper_per.wav", "wb") as wav_file:
    voice.synthesize(text, wav_file)
e=time.time()
e-s


"""##Original piper < microsoft and coqui in quality. ar and eng
"""

!pip install piper-tts
#en
s=time.time()
!echo ' en' | piper \
  --model 'en_US-lessac-medium' \
  --output_file piperpersian.wav
e=time.time()
e-s
'echo "{txt}" | ./piper/piper --model ./model/{model} --output_file {wav_filepath} --sentence_silence 0.3



"""##Alisterta"""
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import sys
import os
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.append('tools')
sys.path.append('network')
# %load_ext autoreload
# %autoreload 2
from hp import HP
from model_graph import model
tf.reset_default_graph()
# tf.compat.v1.reset_default_graph()

# !git clone https://github.com/fastai/courses.git
!git clone https://github.com/AlisterTA/Persian-text-to-speech.git
# %cd /content/Persian-text-to-speech/tools
!cp /content/Persian-text-to-speech/model_graph.py /content/Persian-text-to-speech/tools
gr=model('null','demo')
sentenses=["معلوم بود واقعا به دنبال جوابى براى سوالش نیست"]
gr.predict(sentenses)



"""##coqui-ai(for en and ar)(all voices)(slower than microsoft)"""
!pip install TTS
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(TTS().list_models())
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
s=time.time()
tts.tts_to_file(text=" a few seconds. When it turns off, choose the houses that you remember to be colored", speaker_wav="/content/tts.out.3.wav", language="en", file_path="output.wav")
e=time.time()
e-s



"""##tts karim (very slow)"""
import re
from IPython.display import Audio, display
!pip install -q TTS
!sudo apt-get -y install espeak-ng
from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

#download pretrained models:
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Kamtera/persian-tts-male-vits", filename="checkpoint_77000.pth",local_dir="new")
hf_hub_download(repo_id="Kamtera/persian-tts-male-vits", filename="config.json",local_dir="new")
#or:
hf_hub_download(repo_id="Kamtera/persian-tts-male1-vits", filename="checkpoint_88000.pth",local_dir="new")
hf_hub_download(repo_id="Kamtera/persian-tts-male1-vits", filename="config.json",local_dir="new")

s=time.time()
!tts --text "در این بازی مکعب ها به صورت شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن، خونه هایی که یادت مونده رنگی بودن رو انتخاب کن" \
     --model_path "/content/new/checkpoint_88000.pth" \
     --config_path "/content/new/config.json" \
     --out_path "speech1.wav"
e=time.time()
e-s

# basepath="/content/persian-tts-male1-vits"
basepath="/content/new"
model_path =basepath+"/checkpoint_77000.pth"
config_path =basepath+"/config.json"
# speakers_file_path = # Absolute path to speakers.pth file
# vocoder_path="/checkpoint_105000.pth"#vbasepath+"/checkpoint_127000.pth"
# vocoder_config_path="/content/config-0.json"
synthesizer = Synthesizer(
        model_path,
        config_path,
        None ,#speakers_file_path,
        None ,#language_ids_file_path,
        # vocoder_path ,#vocoder_path,
        # vocoder_config_path ,#vocoder_config_path,
        None,
        None,
        None ,#encoder_path,
        None ,#encoder_config_path,
        None ,#args.use_cuda,
        )
wavs = synthesizer.tts(text)
synthesizer.save_wav(wavs, 'sp.wav')



"""##facebook(wave is better than vits model)"""
# Load model directly
from transformers import AutoTokenizer, AutoModelForTextToWaveform #, VitsModel
import torch
from IPython.display import Audio

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-heb")
model = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-heb")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
# model1 = AutoModelForTextToWaveform.from_pretrained("facebook/mms-tts-eng")
# model2 = VitsModel.from_pretrained("facebook/mms-tts-eng")
# tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

s=time.time()
text = "در این بازی مکعب ها به صورت شبکه ای کنار هم قرار میگیرن. چند ثانیه بعضی از مکعب ها روشن هستن. زمانی که خاموش شدن، خونه هایی که یادت مونده رنگی بودن رو انتخاب کن"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform
e=time.time()
print(e-s)
Audio(output, rate=model.config.sampling_rate)

#per-wave 52.4823477268219
#per-vist 71.8700270652771
#en-wave 44.38858509063721
#en-vist 51.725035190582275


# Implementation - All:

#!pip install faster-whisper
#!pip install ctranslate2==4.4.0
!pip install edge-tts
# ! pip install -U cohere
!sudo apt-get install portaudio19-dev
pip install pyaudio

import pyaudio
import edge_tts
from faster_whisper import WhisperModel,BatchedInferencePipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# # from transformers import pipeline
# import time
# from IPython.display import Audio, display
# import cohere

model_size = "turbo"
# model_size = 'tiny'

# # STT(fast whisper):
model = WhisperModel(model_size, device="cuda", compute_type='int8_float16')
batched_model = BatchedInferencePipeline(model=model)


"""#implement"""

#Emotion detection(bektash from text):
def predict_labels(model, tokenizer, texts):

    labels = ['ANGRY', 'FEAR', 'HAPPY', 'HATE', 'OTHER', 'SAD', 'SURPRISE']
    # Ensure the model is in evaluation mode
    model.eval()

    # Tokenize the texts
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    # Move tensors to the same device as the model
    input_ids = encodings['input_ids'].to(model.device)
    attention_mask = encodings['attention_mask'].to(model.device)

    # Get the model's predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted labels
    _, preds = torch.max(outputs.logits, dim=-1)

    return [labels[item] for item in preds.tolist()]

# context = "امتیاز 0 تا 50 بد است. امتیاز 50 تا 70 متوسط است و امتیاز بالای 70 خوب است."
co = cohere.ClientV2(api_key="")
documents = [
  {
    "data": {
      "text": "درصورتیکه که در مورد بازی بیلدر سوالی انجام شد، در بازی بیلدر، امتیاز 0 تا 50 پایین است، امتیاز بین 50 تا 70 متوسط است و امتیاز بالای 70 خوب است."
    }
  }
]

def res(text):
  input=text
  message = f""" فرض کن تو راهنمای من هستی، به این سوال در حداکثر دو جمله پاسخ بده و درصورتیکه محتوای سوال با محتوای documents مرتبط بود، از محتوای documents برای پاسخ دادن استفاده کن. در غیر این صورت، از دانش خودت استفاده کن: {input}"""
  # Generate the response
  response = co.chat(model="command-r",
                    messages=[{"role": "user", "content": message}],
                    documents=documents,
                    temperature=0.0,
                    max_tokens=200
  )
  return response.message.content[0].text
                  #    messages=[cohere.UserMessage(content=message)])

def tts(text):
  massage="edge-tts --voice fa-IR-DilaraNeural" +"  --text "+ "'"+text+"'"+" --write-media microsoft_per_tts.mp3 --write-subtitles hello.vtt"
  os.system(massage)
  return "TTS operation succesfully Done!"


segments, info = model.transcribe("/content/صدا ۰۴۴.wav", beam_size=5)
text=''
for segment in segments:
  text=text+(segment.text)
predicted_labels = predict_labels(model_text, tokenizer, [text])
# question = text
answer = res(text)
#  [qa_pipeline(question=text, context=context)['answer']]
#FaridNeural
# !edge-tts --voice fa-IR-DilaraNeural --text {answer} --write-media microsoft_per_tts.mp3 --write-subtitles hello.vtt
tts(answer)

print(e-s)
display(Audio(f"/content/microsoft_per_tts.mp3", autoplay=True))