#first method:
import pyaudio
import wave
from faster_whisper import WhisperModel
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 24000 #44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
channels=CHANNELS,
rate=RATE,
input=True,
frames_per_buffer=CHUNK)

print("* recording")

while True:
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    segments, info = model.transcribe(WAVE_OUTPUT_FILENAME, beam_size=5, language="fa", condition_on_previous_text=False)

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


#second method:
from faster_whisper import WhisperModel
import os
import numpy as np
# import pyaudio
from pyaudio import PyAudio,paInt16


model_size = "tiny"
model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=8, local_files_only=True)
pa = PyAudio()
stream =pa.open(format=paInt16, channels=2, rate=48000, input=True, frames_per_buffer=2048)
result = ''
while True:
    audio_data = stream.read(2048)
    a = np.ndarray(buffer=audio_data, dtype=np.int16, shape=(2048,))
    print('start')
    segments, info = model.transcribe(a, language="en")
    for segment in segments:
        result += segment.text
    print(result)