# ----Google Web Speech API-----
#Best in time

#Package Instalation:
! pip install SpeechRecognition

import speech_recognition as sr
import time
sr.__version__

# Initialize the recognizer
#recognizer = sr.Recognizer()
s=time.time()
r = sr.Recognizer()
sound = sr.AudioFile('/content/test1.wav')
with sound as source:
    audio = r.record(source)
    try:
        # using google speech recognition
        print("Text: "+r.recognize_google(audio, language='ar-AR'))
    except:
         print("Sorry, I did not get that")#print(r.recognize_google(audio))
e=time.time()
print(e-s)
