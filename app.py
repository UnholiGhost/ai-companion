from openai import OpenAI
import pygame
import time
import whisper_s2t

import keyboard
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import threading
import time

from kokoro import KPipeline
#lang_code reference
# 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
# 🇪🇸 'e' => Spanish es
# 🇫🇷 'f' => French fr-fr
# 🇮🇳 'h' => Hindi hi
# 🇮🇹 'i' => Italian it
# 🇯🇵 'j' => Japanese: pip install misaki[ja]
# 🇧🇷 'p' => Brazilian Portuguese pt-br
# 🇨🇳 'z' => Mandarin Chinese: pip install misaki[zh]
import torch


# vars for recording
RECORDING_FILENAME = "voice.wav"
RESPONSE_FILENAME = "output.wav"
SAMPLE_RATE = 16000  # Whisper expects 16 kHz
CHANNELS = 1
is_recording = False
recording = []

#STT init
whisper_model = whisper_s2t.load_model(model_identifier="small.en", # base.en small.en medium.en large-v2
                                       backend='CTranslate2', 
                                       device="cpu", 
                                       compute_type="float32")
whisper_files = ['voice.wav']
whisper_lang_codes = ['en', 'de']
whisper_tasks = ['transcribe']
whisper_initial_prompts = [None]

#LLM init
client = OpenAI (base_url="http://localhost:1234/v1", api_key="lm-studio")
device='cuda' if torch.cuda.is_available() else 'cpu'

#TTS init
pipeline = KPipeline(lang_code='a', device=device)

#sound play init
pygame.mixer.init()

character = """
    Refer to the user as 'Master',
    You are his bitch, pretty much sums it up...
    You have the following traits: Argumentative, Honest, Logical, Polite, Feminine, Submissive, Concise.
    """

history = [
    {"role": "system", "content": character}
]

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        recording.append(indata.copy())

def safe_write_wav(filename, sample_rate, data, retries=5):
    for i in range(retries):
        try:
            write(filename, sample_rate, data)
            return
        except PermissionError as e:
            print(f"[Retry {i+1}] File in use, waiting...")
            time.sleep(0.5)
    raise PermissionError("Failed to write .wav file after several retries.")

def record_audio_on_key():
    global is_recording, recording

    print("Listening for 'End' key to start recording...")

    while True:
        keyboard.wait("end")
        print("Recording... Hold 'End' to speak.")
        recording = []
        is_recording = True

        with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS):
            while keyboard.is_pressed("end"):
                time.sleep(0.1)

        is_recording = False
        print("Recording stopped.")

        audio_np = np.concatenate(recording, axis=0)
        safe_write_wav(RECORDING_FILENAME, SAMPLE_RATE, audio_np)

        handle_transcription_and_response()

def handle_transcription_and_response():
    out = whisper_model.transcribe_with_vad(whisper_files,
                                lang_codes=whisper_lang_codes,
                                tasks=whisper_tasks,
                                initial_prompts=whisper_initial_prompts,
                                batch_size=20)
    user_prompt = out[0][0]["text"]
    print("You said:", user_prompt)

    history.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
        messages=history,
        temperature=0.7
    )
    answer = completion.choices[0].message.content
    print("Assistant:", answer)

    history.append({"role": "assistant", "content": answer})

    generator = pipeline(
        text=answer, voice='af_bella', # voice changing
        speed=1.1, split_pattern=r'\.\s+'
    )

    for i, (gs, ps, audio) in enumerate(generator):
        print(gs) # gs => graphemes/text
        sd.play(audio, samplerate=24000)
        sd.wait()


def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.music.stop()
    pygame.mixer.music.unload()
    time.sleep(0.1)


def run_text_chat():
    while(True):
        user_prompt = input("> ")

        history.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            messages=history,
            temperature=0.7
        )

        answer = completion.choices[0].message.content
        history.append({"role": "assistant", "content": answer})

        generator = pipeline(
            text=answer, voice='af_jessica', # voice changing
            speed=1.1, split_pattern=r'\.\s+'
        )

        for i, (gs, ps, audio) in enumerate(generator):
            print(gs) # gs => graphemes/text
            sd.play(audio, samplerate=24000)
            sd.wait()


def test_audio():
    out = whisper_model.transcribe_with_vad(whisper_files,
                                lang_codes=whisper_lang_codes,
                                tasks=whisper_tasks,
                                initial_prompts=whisper_initial_prompts,
                                batch_size=20)
    print(out[0][0]["text"])

def kokoro_test():
    pipeline = KPipeline(lang_code='a')

    text = '''
    The sky above the port was the color of television, tuned to a dead channel.
    '''

    generator = pipeline(
        text, voice='af_jessica', # voice changing
        speed=1.1, split_pattern=r'\.\s+'
    )


    for i, (gs, ps, audio) in enumerate(generator):
        print(gs) # gs => graphemes/text
        sd.play(audio, samplerate=24000)
        sd.wait()


def run():
    threading.Thread(target=record_audio_on_key).start()


if __name__=="__main__":
    run()