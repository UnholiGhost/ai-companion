from openai import OpenAI
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


# recording init
RECORDING_FILENAME = "voice.wav"
RESPONSE_FILENAME = "output.wav"
SAMPLE_RATE = 16000  # Whisper expects 16 kHz
CHANNELS = 1
recording = []
recording_event = threading.Event()
assistant_playing = threading.Event()

#STT init
whisper_model = whisper_s2t.load_model(model_identifier="base.en", 
                                       backend='CTranslate2', 
                                       device="cpu", 
                                       compute_type="float32")
whisper_files = ['voice.wav']
whisper_lang_codes = ['en']
whisper_tasks = ['transcribe']
whisper_initial_prompts = [None]

#LLM init
client = OpenAI (base_url="http://localhost:1234/v1", api_key="lm-studio")
device='cuda' if torch.cuda.is_available() else 'cpu'

#TTS init
pipeline = KPipeline(lang_code='a', device=device)

character = """
    Refer to the user as 'Master',
    You are his bitch, pretty much sums it up...
    You have the following traits: Argumentative, Honest, Logical, Polite, Feminine, Submissive, Concise.
    When speaking, try to resemble real-life human speech. You don't use emotes and stage directions.
    """

history = [
    {"role": "system", "content": character}
]

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio callback warning] {status}")
    if recording_event.is_set():
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

def on_key_event(e):
    global recording

    if e.name == 'esc' and e.event_type == 'down':
        on_exit()

    if e.name == 'end':
        if e.event_type == 'down':
            if not recording_event.is_set():
                print("Recording started...")
                #  INTERRUPT 
                if assistant_playing.is_set():
                    assistant_playing.clear() 
                recording.clear()
                recording_event.set()
                threading.Thread(target=record_audio).start()
        elif e.event_type == 'up':
            print("Recording stopped.")
            recording_event.clear()


def record_audio():
    global recording

    with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=CHANNELS):
        while recording_event.is_set():
            time.sleep(0.1)

    if recording:
        audio_np = np.concatenate(recording, axis=0)
        safe_write_wav(RECORDING_FILENAME, SAMPLE_RATE, audio_np)
        handle_transcription_and_response()
    else:
        print("No audio recorded.")


def handle_transcription_and_response():
    out = whisper_model.transcribe_with_vad(whisper_files,
                                lang_codes=whisper_lang_codes,
                                tasks=whisper_tasks,
                                initial_prompts=whisper_initial_prompts,
                                batch_size=20)
    user_prompt = out[0][0]["text"]
    print("You said:", user_prompt)
    print()

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

    unspoken = []
    assistant_playing.set()

    for i, (gs, ps, audio) in enumerate(generator):
        if not assistant_playing.is_set():
            unspoken.append(gs)
            continue

        print(gs)
        sd.play(audio, samplerate=24000)
        sd.wait()

    assistant_playing.clear()

    # if interrupted inform assistant
    if unspoken:
        interruption_note = (
            "Some of the assistant's sentences were not spoken due to user interruption.\n\n"
            "Unspoken sentences:\n" + "\n".join(unspoken)
        )
        history.append({"role": "system", "content": interruption_note})



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

def on_exit():
    print("\nExiting...")
    recording_event.clear()
    sd.stop()
    keyboard.unhook_all()
    exit(0)


#################
##### TEST ######
#################

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

#################
###### END ######
#################


def run():
    keyboard.hook(on_key_event)
    print("Press and hold 'End' to record. Release to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        on_exit()


if __name__=="__main__":
    run()