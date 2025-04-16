#Audio Transcription:

#installing packages:
# !pip install faster-whisper
# !pip install pydub

from faster_whisper import WhisperModel ,BatchedInferencePipeline
import pandas as pd
import librosa
import soundfile as sf
from pydub import AudioSegment
import time

model_size = "large-v3"
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

data=pd.DataFrame(columns=['audio','text','s_time','e_time'])
data.info()

path='/content/drive/MyDrive/20152.wav'

s=time.time()
# segments, info = model.transcribe("test_stt2.wav", beam_size=5)
# segments,info = model.transcribe(path,language='ar',beam_size=10)#,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=500))#, beam_size=5)
segments, info = batched_model.transcribe(path,language='ar', batch_size=16,chunk_length=8) #,vad_filter=True,vad_parameters=dict(min_silence_duration_ms=200))

i=0
for segment in segments:
  data.loc[i, 's_time'] =segment.start
  data.loc[i,'e_time']=segment.end
  data.loc[i,'text']=segment.text
  i=i+1
e=time.time()
print(e-s)
data.to_excel('/content/drive/MyDrive/20152.xlsx')

# Audio chunking:
audio = AudioSegment.from_wav(path)
for i in range(0,len(data)):
  chunk = audio[(data['s_time'].iloc[i])*1000:(data['e_time'].iloc[i])*1000]
  chunk.export('/content/drive/MyDrive/20152/'+str(i)+'.wav', format='wav')