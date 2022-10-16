import whisper
import soundcard as sc
import threading
import queue
import numpy as np
import argparse

SAMPLE_RATE = 16000
INTERVAL = 3
BUFFER_SIZE = 4096

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='base')
args = parser.parse_args()

print('Loading model...')
model = whisper.load_model(args.model)
print('Done')

q = queue.Queue()
b = np.ones(100) / 100

options = whisper.DecodingOptions()

def recognize():
    while True:
        audio = q.get()
        if (audio ** 2).max() > 0.001:
            audio = whisper.pad_or_trim(audio)

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)

            # decode the audio
            result = whisper.decode(model, mel, options)

            # print the recognized text
            print(f'{max(probs, key=probs.get)}: {result.text}')


th_recognize = threading.Thread(target=recognize, daemon=True)
th_recognize.start()

# start recording
with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
    audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
    n = 0
    while True:
        while n < SAMPLE_RATE * INTERVAL:
            data = mic.record(BUFFER_SIZE)
            audio[n:n+len(data)] = data.reshape(-1)
            n += len(data)

        # find silent periods
        m = n * 4 // 5
        vol = np.convolve(audio[m:n] ** 2, b, 'same')
        m += vol.argmin()
        q.put(audio[:m])

        audio_prev = audio
        audio = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
        audio[:n-m] = audio_prev[m:n]
        n = n-m
