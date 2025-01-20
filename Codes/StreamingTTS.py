#first model(it has stops):
import edge_tts
import pyaudio
from io import BytesIO
from pydub import AudioSegment
import cohere

def main(TEXT) -> None:
    communicator = edge_tts.Communicate(TEXT, VOICE)
    audio_chunks = []

    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    for chunk in communicator.stream_sync():
        if chunk["type"] == "audio" and chunk["data"]:
            audio_chunks.append(chunk["data"])
            if len(audio_chunks) >= CHUNK_SIZE:
                play_audio_chunks(audio_chunks, audio_stream)
                audio_chunks.clear()

    play_audio_chunks(audio_chunks, audio_stream)

    audio_stream.stop_stream()
    audio_stream.close()
    pyaudio_instance.terminate()

def play_audio_chunks(chunks: list[bytes], stream: pyaudio.Stream) -> None:
    stream.write(AudioSegment.from_mp3(BytesIO(b''.join(chunks))).raw_data)

main(TEXT)

#second model:
import asyncio
import edge_tts
import pygame
from io import BytesIO


async def main():
    pygame.mixer.init()

    tts = edge_tts.Communicate("فرآیند داوری آثار دخیل کرد و این موضوع به شور و اشتیاق جشنواره افزود.", "fa-IR-DilaraNeural")

    mp3_data = bytearray()
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            mp3_data.extend(chunk["data"])

    mp3_stream = BytesIO(mp3_data)
    pygame.mixer.music.load(mp3_stream)

    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(0.1)

asyncio.run(main())

#third method(faster but less setting):
import edge_tts
import pyaudio
from io import BytesIO
from pydub import AudioSegment

TEXT = "فرآیند داوری آثار دخیل کرد و این موضوع به شور و اشتیاق جشنواره افزود."
VOICE = "fa-IR-DilaraNeural"#"en-US-AndrewMultilingualNeural"
CHUNK_SIZE = 20

def main() -> None:
    communicator = edge_tts.Communicate(TEXT, VOICE)
    audio_chunks = []

    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = pyaudio_instance.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    for chunk in communicator.stream_sync():
        if chunk["type"] == "audio" and chunk["data"]:
            audio_chunks.append(chunk["data"])
            if len(audio_chunks) >= CHUNK_SIZE:
                play_audio_chunks(audio_chunks, audio_stream)
                audio_chunks.clear()

    play_audio_chunks(audio_chunks, audio_stream)

    audio_stream.stop_stream()
    audio_stream.close()
    pyaudio_instance.terminate()

def play_audio_chunks(chunks: list[bytes], stream: pyaudio.Stream) -> None:
    stream.write(AudioSegment.from_mp3(BytesIO(b''.join(chunks))).raw_data)

main()
