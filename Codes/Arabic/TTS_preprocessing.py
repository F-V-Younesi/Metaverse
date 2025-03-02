#dataset preprocessing:

##download wav format in mono channel and Sample rate=48000Hz from youtube:
!pip install yt-dlp
!pip install pydub

from yt_dlp import YoutubeDL
from pydub import AudioSegment
import os

def download_youtube_audio(url, output_path="audio.wav"):

    try:
        ydl_opts = {
            'format': 'bestaudio/best', 
            'outtmpl': 'temp_audio.%(ext)s',
            'postprocessors': [{     
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Audio downloaded successfully")

        audio = AudioSegment.from_file("temp_audio.wav")
        audio = audio.set_frame_rate(48000)
        audio.export(output_path, format="wav")
        os.remove("temp_audio.wav")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example
youtube_url = "https://www.youtube.com/watch?v=HVEem0AnZ6k"
download_youtube_audio_first_5_minutes(youtube_url, output_path="2015speech.wav")


##Chunking Audio:

import librosa
d,s=librosa.load('2015speech.wav',sr=None)
print(s)

length=len(y)/60/sr
n=(float(length/10)+1)*10

listi=[]
for i in range(0,n):
  if i%10==0:
    listi.append(i)

#save 10 min chunks of downloaded file in sr=48000:
for i in listi:
  if i!=n-10:
    start_time = i*60.0  # Start at 10 seconds
    end_time = (i+10)*60.0

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    audio_segment = y[start_sample:end_sample]

    output_path = '2015_'+str(i)+'_'+str(i+10)+'.wav'
    sf.write(output_path, audio_segment, sr)
  if i==n-10:
    start_time = i*60.0  # Start at 10 seconds

    start_sample = int(start_time * sr)

    audio_segment = y[start_sample:]

    output_path = '2015_'+str(i)+'_'+str(i+10)+'.wav'
    sf.write(output_path, audio_segment, sr)


#DeepFilterNet:
pip install deepfilternet
for i in listi:
    path = '2015_'+str(i)+'_'+str(i+10)+'.wav'
    !deepFilter $path #--output-dir '/deeplfilter'


###resampling to 22050
for i in listi:
  path = '2015_'+str(i)+'_'+str(i+10)+'_DeepFilterNet3'+'.wav'
  d,s=librosa.load(path,sr=None)
  d2=librosa.resample(d,orig_sr=s,target_sr=22050)
  path2 = '2015_'+str(i)+'_'+str(i+10)+'_DeepFilterNet3'+'_resample'+'.wav'
  sf.write(path2,d2,22050)

