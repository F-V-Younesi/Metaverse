##speech_rec(google model)

! pip install SpeechRecognition

import speech_recognition as sr
import time
sr.__version__


# Initialize the recognizer
recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    try:
        # Load audio file
        with sr.AudioFile(file_path) as source:
            # Adjust for ambient noise and record
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

        # Recognize (convert from speech to text)
        text = recognizer.recognize_google(audio)
        return text

    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Provide the path to WAV file
file_path = "1.wav"
transcribed_text = transcribe_audio(file_path)
print("Transcription:", transcribed_text)

#or use :Google Web Speech API (default)
r = sr.Recognizer()
harvard = sr.AudioFile('1.wav')
with harvard as source:
    audio = r.record(source)
    try:
        # using google speech recognition
        print("Text: "+r.recognize_google(audio, language='ar-AR'))
    except:
         print("Sorry, I did not get that")#print(r.recognize_google(audio))


##assembly ai(nano model) slow and not acc

!pip install assemblyai

import assemblyai as aai
import time
aai.settings.api_key = ""

transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(
    language_code="ar",
    speech_model=aai.SpeechModel.nano 
)
transcript = transcriber.transcribe("1.wav", config=config)
print(transcript.text)


##speechmatics

!pip install speechmatics-python
!sudo apt-get install portaudio19-dev
!pip install pyaudio

!speechmatics config set --auth-token '' # --rt-url wss://eu2.rt.speechmatics.com/v2 --batch-url https://asr.api.speechmatics.com/v2
# !speechmatics config unset --rt-url --batch-url

#file input with cli:
!speechmatics transcribe --lang ar  aud.wav

#Batch Client Usage %%%%Slooow%%%

from speechmatics.models import ConnectionSettings, BatchTranscriptionConfig
from speechmatics.batch_client import BatchClient
from httpx import HTTPStatusError

API_KEY = ""
PATH_TO_FILE = "1.wav"
LANGUAGE = "ar"

# Open the client using a context manager
with BatchClient(API_KEY) as client:
    try:
        job_id = client.submit_job(PATH_TO_FILE, BatchTranscriptionConfig(LANGUAGE))
        # print(f'job {job_id} submitted successfully, waiting for transcript')

        transcript = client.wait_for_completion(job_id, transcription_format='txt')
        print(transcript)
    except HTTPStatusError as e:
        if e.response.status_code == 401:
            print('Invalid API key - Check API_KEY ')
        elif e.response.status_code == 400:
            print(e.response.json()['detail'])
        else:
            raise e


##stream stt:
!sudo apt-get install portaudio19-dev
!pip install pyaudio
!pip install websockets

import threading
import websocket
import pyaudio
import json


# Audio stream parameters
CHUNK_SIZE = 1024
SAMPLE_RATE = 16000

def on_message(ws, message):
    result = json.loads(message)
    if 'results' in result:
        for r in result['results']:
            if 'text' in r:
                print(f"Transcription: {r['text']}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws):
    print("Connection closed")

def on_open(ws):
    def run():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
        print("Starting transcription (press Ctrl+C to stop):")

        try:
            while True:
                audio_data = stream.read(CHUNK_SIZE)
                ws.send(audio_data, opcode=websocket.ABNF.OPCODE_BINARY)
        except KeyboardInterrupt:
            print("\nTranscription stopped.")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    threading.Thread(target=run).start()

websocket_url = f"wss://api.speechmatics.com/v2/rt/transcription?lang={LANGUAGE}&domain=default&access_token={API_KEY}"
ws = websocket.WebSocketApp(websocket_url,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)
ws.on_open = on_open
ws.run_forever()


##klaam(use directly the model=1.3 g),only 8 s, low acc) check updates foe longer files

!git clone https://github.com/ARBML/klaam
%cd klaam

%%capture
!pip install -r requirements.txt
!pip install mishkal
!pip install unidecode

from klaam import SpeechRecognition
model = SpeechRecognition(lang = 'msa')
t=model.transcribe('aud.wav')
print(t)


##faster whisper
!pip install faster-whisper
!pip install ctranslate2==4.4.0

from faster_whisper import WhisperModel

model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("1.wav", beam_size=5)
text=''
for segment in segments:
  text=text+(segment.text)


##whisperX(model-3.1G)

%%capture
pip install whisperx

import whisperx
model = whisperx.load_model("large-v2",device='cuda',language="ar")
audio_file = "/content/1.wav"
resultx = model.transcribe(audio_file)

for segment in resultx['segments']:
    print(f"{segment['start']} - {segment['end']}: {segment['text']}")


#not worked/paid models:

##pocketsphinx (not works for ar)

!pip install pocketsphinx
from pocketsphinx import AudioFile

for phrase in AudioFile("/content/mp3-output-ttsfree(dot)com (1).wav"): print(phrase)

! pip install SpeechRecognition

import os
import speech_recognition as sr

# model_path = "/path/to/pocketsphinx/models"
# language_model = os.path.join(model_path, "ar-ar.cd_cont_5000")
# dictionary = os.path.join(model_path, "ar-ar.dic")

# Initialize the recognizer
recognizer = sr.Recognizer()

def transcribe_audio(file_path):
    try:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
        text = recognizer.recognize_sphinx(audio, language="ar-ar")#, show_all=False, model=language_model, dictionary=dictionary)
        return text

    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Provide the path to your WAV file
file_path = "mp3-output-ttsfree(dot)com (1).wav"
transcribed_text = transcribe_audio(file_path)

print("Transcription:", transcribed_text)


##google-cloud-speech(need cloud acount)

pip install google-cloud-speech

from google.cloud import speech_v1p1beta1 as speech
import io

def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ar-AR",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))

# Replace 'path_to_your_audio_file.wav' with the path to your audio file
transcribe_audio("/content/mp3-output-ttsfree(dot)com (1).wav")


##watson(not worked in colab)

!sudo apt-get install portaudio19-dev
!pip install pyaudio
!pip install watson-developer-cloud
!pip install ibm_watson

import pyaudio
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from threading import Thread
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

try:
    from Queue import Queue, Full
except ImportError:
    from queue import Queue, Full

CHUNK = 1024
BUF_MAX_SIZE = CHUNK * 10
q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK)))
audio_source = AudioSource(q, True, True)

authenticator = IAMAuthenticator('api_key')
speech_to_text = SpeechToTextV1(authenticator=authenticator)

# define callback for the speech to text service
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        print(hypothesis)

    def on_data(self, data):
        print(data)

    def on_close(self):
        print("Connection closed")

def recognize_using_weboscket(*args):
    mycallback = MyRecognizeCallback()
    speech_to_text.recognize_using_websocket(audio=audio_source,
                                             content_type='audio/l16; rate=44100',
                                             recognize_callback=mycallback,
                                             interim_results=True)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def pyaudio_callback(in_data, frame_count, time_info, status):
    try:
        q.put(in_data)
    except Full:
        pass # discard
    return (None, pyaudio.paContinue)

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=pyaudio_callback,
    start=False
)
print("Enter CTRL+C to end recording...")
stream.start_stream()
try:
    recognize_thread = Thread(target=recognize_using_weboscket, args=())
    recognize_thread.start()

    while True:
        pass
except KeyboardInterrupt:
    # stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_source.completed_recording()



##RevAi(not respond)

pip install rev_ai

from rev_ai import apiclient
import json

api_key = ''
client = apiclient.RevAiAPIClient(api_key)
audio_file = '1.wav'
job = client.submit_job_local_file(audio_file, language='ar')

while True:

    # Obtains details of a job in json format
    job_details = client.get_job_details(job.id)
    # details_dict = json.loads(job_details)
    status = job_details.status #details_dict["status"]

    # Checks if the job has been transcribed
    if status == "in_progress":
        time.sleep(5)
        continue
    elif status == "failure":

        print('failed')

        break
    break
transcript_text = client.get_transcript_text(job.id)
client.delete_job(job.id)
print(transcript_text)


##otter.ai(not resopnd)

pip install requests

import requests
email = ''
password = ''
auth_url = 'https://otter.ai/oauth/token'
auth_data = {
    'email': email,
    'password': password
}
response = requests.post(auth_url, data=auth_data)
token = response.json()['access_token']
audio_file = '1.wav'
upload_url = 'https://otter.ai/api/v1/import/file'
headers = {
    'Authorization': f'Bearer {token}'
}
files = {
    'file': open(audio_file, 'rb')
}
response = requests.post(upload_url, headers=headers, files=files)
transcription_id = response.json()['transcription_id']
result_url = f'https://otter.ai/api/v1/transcriptions/{transcription_id}/content'
response = requests.get(result_url, headers=headers)
transcription = response.json()['transcript']
print(transcription)


