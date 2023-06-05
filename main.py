import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from fastapi import FastAPI
from scipy.io import wavfile
from scipy import signal
from googletrans import Translator
import os
import subprocess

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()
translator = Translator()


def load_model(language):
    if language == "hi":
        processor = Wav2Vec2Processor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
        model = Wav2Vec2ForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
    elif language == "en":
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    elif language == "mr":
        processor = Wav2Vec2Processor.from_pretrained('tanmaylaud/wav2vec2-large-xlsr-hindi-marathi')
        model = Wav2Vec2ForCTC.from_pretrained('tanmaylaud/wav2vec2-large-xlsr-hindi-marathi')
    else:
        raise ValueError("Invalid language specified")

    return processor, model

def transcribe_audio(audio_file, language):
    processor, model = load_model(language)
    target_duration = 20  
    target_sample_rate = 16000

    sample_rate, audio_input = wavfile.read(audio_file)
    num_samples = int(sample_rate * target_duration)
    os.makedirs('./tmp', exist_ok=True)
    num_files = len(audio_input) // num_samples
    
    if num_files == 0:
        num_files =1 
    transcriptions_org = []
    transcriptions_en =[]
    for i in range(num_files):
        # Calculate the start and end indices for the smaller audio file
        start_index = i * num_samples
        end_index = (i + 1) * num_samples
        
        # Extract the smaller audio file
        smaller_audio = audio_input[start_index:end_index]
        
        # Save the smaller audio file
        smaller_audio_file = os.path.join('./tmp', f"audio_{i+1}.wav")
        wavfile.write(smaller_audio_file, sample_rate, smaller_audio)
        
        s_sample_rate, s_audio_input = wavfile.read(smaller_audio_file)
        if sample_rate != target_sample_rate:
            s_audio_input = signal.resample(s_audio_input, int(len(s_audio_input) * target_sample_rate / s_sample_rate))

        input_values = processor(s_audio_input, sampling_rate=target_sample_rate, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
        
        transcriptions_org.append(transcription)
        if language != "en":
            # Translate to English using Google Translate
            translation = translator.translate(transcription,src=language,dest='en')
            transcription = translation.text
   

        transcriptions_en.append(transcription)
    print(transcriptions_org)

    return " ".join(transcriptions_en)

 


@app.post("/transcribe/{language}")
async def transcribe(request: dict, language: str):
    if language not in ['hi','mr','en']:
        return "Invalid Input Language"
    audio_url = request["audio_url"]
    mode ='test'

    # print(audio_url)
    if mode =='prod':
        subprocess.call(['curl', '-o', 'audio.wav', audio_url])
        transcription = transcribe_audio('audio.wav', language=language)
        return {"transcription": transcription}
    else:
        transcription = transcribe_audio(audio_url, language=language)
        return {"transcription": transcription}