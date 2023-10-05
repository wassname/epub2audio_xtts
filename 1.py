import tika
from tika import parser

import argparse

# Get the command line arguments
parser2 = argparse.ArgumentParser()
parser2.add_argument('--epub', type=str, default='data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub',
                    help='PDF file to read')
parser2.add_argument('--out', type=str, default='out', help='Output folder')
parser2.add_argument('--limit', type=int, default=650,
                    help='Maximum number of characters to synthesize at once')
parser2.add_argument('--model', type=str, 
                    default="tts_models/multilingual/multi-dataset/xtts_v1",
                    # default='facebook/fastspeech2-en-ljspeech',
                    help='fairseq model to use from HuggingFace Hub')
parser2.add_argument('--vocoder', type=str, default='hifigan',
                    help='Vocoder to use from the model')
parser2.add_argument('--speaker', type=int, default=0,
                    help='Speaker to use from the model')
args = parser2.parse_args()

fileIn = args.epub

parsed = parser.from_file(fileIn)
content = parsed["content"]

# generate speech by cloning a voice using default settings
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1", gpu=True)
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="example.wav",
                language="en")

