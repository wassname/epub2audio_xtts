"""modified from https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c"""
from tika import parser

import torchaudio
import argparse
from sanitize_filename import sanitize
import re
from pathlib import Path

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

import torch
import json
from dataclasses import dataclass
# import pysbd
from typing import List
from loguru import logger
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


def split_into_sentences(text) -> List[str]:        
    limit = 400
    chunk_limit = limit//3
    splitter = RecursiveCharacterTextSplitter(
        length_function=lambda x: len(tokenizer.encode(x, lang="en")),
        chunk_size=chunk_limit,
        chunk_overlap=0,
        keep_separator=True,
        strip_whitespace=True,
        separators=[
                    "\n\n", "\n", "\xa0", '<div>', '<p>', '<br>', "\r", ".",  "!", "?", 
            '"', "'", "‘", "’", "“", "”", "„", "‟",  
            "(", ")", "[", "]", "{", "}", 
            "…", ":", ";", "—",
            " ", '' # these ensure that there is always something to split by so chunks are always at limit
    ],
    )
    texts = splitter.split_text(text)
    ls = [splitter._length_function(x) for x in texts]
    logger.debug(f'split lengths {ls}. max={max(ls)} chunk_limit={chunk_limit}')
    assert all([l<=limit for l in ls]), 'all senteces should be below limit'
    return texts

root_dir = Path(__file__).parent.parent.absolute()


# Get the command line arguments
parser2 = argparse.ArgumentParser()
parser2.add_argument('--epub', type=Path, 
                    #  default='data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub',
                     default='data/golden_saying_of_epictetus.epub',
                    help='PDF file to read')
parser2.add_argument('-o', '--out', type=Path, default=None, help='Output folder')
parser2.add_argument('-f', '--force', action='store_true', default=False, help='Overwrite')
parser2.add_argument('-t', '--test', action='store_true', default=False, help='Overwrite')
parser2.add_argument('-l', '--limit', type=int, default=400,
                    help='Maximum number of characters to synthesize at once')
parser2.add_argument('-m', '--model', type=str, 
                    default="tts_models/multilingual/multi-dataset/xtts_v1",
                    # default='facebook/fastspeech2-en-ljspeech',
                    help='fairseq model to use from HuggingFace Hub')
parser2.add_argument('-s', '--speaker', type=Path, default=root_dir / "data/speakers/donaldrobertson.wav",
                    help='Speaker wav to use from the model')
args = parser2.parse_args()

if args.out is None:
    from datetime import datetime
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
    args.out = root_dir / 'out' / (sanitize(args.epub.stem).replace(' ', '_').lower() + timestamp)

# load epib
parsed = parser.from_file(str(args.epub))
text = parsed["content"]
if args.test:
    text = text[:1000]


# make output directory
out_dir = Path(args.out)
if out_dir.exists():
    if not args.force:
        logger.warning('Output folder already exists. Use -f to overwrite.')
        exit(1)
    else:
        for f in out_dir.glob('*'):
            f.unlink()
        out_dir.rmdir()
out_dir.mkdir()
logger.info(f'Output folder: {out_dir}')

@dataclass
class Writer:
    out_dir: Path
    # tts: TTS
    
    def __post_init__(self):
        self.m3u = open(self.out_dir / 'playlist.m3u', 'w')
        self.m3u.write('#EXTM3U\n')
        self.chapter = 1

    def write_chapter(self, waveforms):
        wav_f = out_dir / f'{self.chapter}.wav'
        # self.tts.synthesizer.save_wav(wav=waveforms, path=wav_f)
        torchaudio.save(wav_f, waveforms.squeeze(0).cpu(), 24000)
        self.m3u.write(f'{wav_f}\n')
        self.chapter += 1

    def close(self):
        self.m3u.close()



# write metadata to dir
from json_tricks import dump, dumps, load, loads, strip_comments
f_metadata = out_dir / 'metadata.json'
with open(f_metadata, 'w') as fo:
    dump(dict(
        epub_metadata=parsed['metadata'],
        args=args.__dict__,
        
    ), fo, indent=4)

# load model
use_cuda = False if args.test else torch.cuda.is_available()
logger.info(f'use_cuda {use_cuda}')
# tts = TTS2(args.model, gpu=use_cuda, progress_bar=True)
# voice_samples, conditioning_latents = load_voice(voice)
tts = TextToSpeech(use_deepspeed=True, kv_cache=True, half=True)
writer = Writer(out_dir, tts)
clips_paths = [args.speaker]
reference_clips = [load_audio(p, 22050) for p in clips_paths]

# split text
text = [text] #limit_text_len(text, max_len=args.limit, lang='en')
waveforms = []
for i, t in enumerate(text):
    t = t.replace('\n', ' ').strip()
    # Skip empty text
    if t == None or t == '':
        continue

    # check if contains words or numbers
    if not re.search('[a-zA-Z0-9]', t):
        logger.debug(f'Skipping text without words or numbers {t}')
        continue
    logger.debug(f'current sentence {t}')
    
    # https://github.com/neonbjb/tortoise-tts 
    # https://github.com/neonbjb/tortoise-tts/blob/main/tortoise_tts.ipynb
    # ultra_fast fast standard high_quality
    wav = tts.tts_with_preset("your text here", voice_samples=reference_clips, preset='fast')
    
    # wav = tts.tts(text=t, language="en", speaker_wav=args.speaker)
    waveforms += wav
    
    if len(waveforms) > 10000000//4:  # ~20G
        wav_f = out_dir / f'{i}.wav'
        logger.warning(f"wrote chapter {wav_f}")
        writer.write_chapter(waveforms, wav_f)
        waveforms = []
        
if len(waveforms):  
    writer.write_chapter(waveforms)
writer.close()
