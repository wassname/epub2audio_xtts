"""modified from https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c"""
import tika
from tika import parser

import argparse
from sanitize_filename import sanitize
import re
from pathlib import Path
from TTS.api import TTS
import pdb
import torch
import json
from dataclasses import dataclass
# import pysbd
from typing import List
import tiktoken

from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text.characters import Graphemes, IPAPhonemes
# tokenizer = TTSTokenizer(use_phonemes=False, characters=Graphemes())
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# TTS.tts.layers.xtts.tokenizer.VoiceBpeTokenizer
# encoding = VoiceBpeTokenizer(vocab_file='/home/wassname/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1/vocab.json')
# encoding = tiktoken.get_encoding("cl100k_base")

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter



class RecursiveCharacterTextSplitter2(RecursiveCharacterTextSplitter):
    

def split_string_with_limit(text: str, limit: int, encoding: tiktoken.Encoding):
    # FIXME: this returns decoded text e.g. 'a short guide to the inner citadel a short guide to the inner citadel on pierre hadot’s classic analysis of marcus aurelius’ r '
    # instead use langchain's https://github.com/langchain-ai/langchain/blob/57ade13b2b9c75d2967eb13c91417c356e2c805d/libs/langchain/langchain/text_splitter.py#L226
    """Split a string into parts of given size without breaking words.
    
    Args:
        text (str): Text to split.
        limit (int): Maximum number of tokens per part.
        encoding (tiktoken.Encoding): Encoding to use for tokenization.
        
    Returns:
        list[str]: List of text parts.
        
    modified from https://gist.github.com/izikeros/17d9c8ab644bd2762acf6b19dd0cea39
        
    """
    splitter = TextSplitter(length_function=lambda x:len(encoding(x)))
    return splitter.split_text(text, limit)
    
    tokens = encoding.encode(text, "en")
    parts = []
    text_parts = []
    current_part = []
    current_count = 0

    for token in tokens:
        current_part.append(token)
        current_count += 1

        if current_count >= limit:
            parts.append(current_part)
            current_part = []
            current_count = 0

    if current_part:
        parts.append(current_part)

    # Convert the tokenized parts back to text
    for part in parts:
        text = [
            encoding.decode([token]) #.decode("utf-8", errors="replace")
            for token in part
        ]
        text_parts.append("".join(text))

    return text_parts

from TTS.utils.synthesizer import Synthesizer
class Synthesizer2(Synthesizer):
    def split_into_sentences(self, text) -> List[str]:
        # TODO the best split is this https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#TextSplitter?
        # Or use tikktoken?
        
        segs = split_string_with_limit(text, 400, self.tts_model.tokenizer)
        # segs = self.seg.segment(text)
        # segs = limit_len_text(segs, 400) # otherwise we seem to get positional emeddingerrors
        # pdb.set_trace()
        return segs

class TTS2(TTS):
    """modify this so that each sentace is below min chars."""
    
    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """
        self.synthesizer = None
        self.csapi = None
        self.model_name = model_name

        if "coqui_studio" in model_name:
            self.csapi = CS_API()
        else:
            model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(
                model_name
            )

            # init synthesizer
            # None values are fetch from the model
            self.synthesizer = Synthesizer2(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config_path,
                encoder_checkpoint=None,
                encoder_config=None,
                model_dir=model_dir,
                use_cuda=gpu,
            )



# def limit_text_len(text, max_len=200, seps=['.', ',', '\n', ' '], lang = "en") -> list:
#     seg = pysbd.Segmenter(language=lang, clean=True)
#     return seg.segment(text)
#     self.voice_converter.vs_model
    
# def limit_len_text(text: list, max_len=400, seps=['.', ',', '\n', ' '], lang = "en") -> list: 
#     """If the some element of text is too long, divide it in smaller parts using the separators."""
#     return_text = []
#     i = 0
#     for t in text:
#         if len(t) > max_len:
#             splited = t.split(seps[0])

#             # Restore the separators
#             for j in range(0, len(splited)-1):
#                 splited[j] = splited[j] + seps[0]

#             # If the split is not good enough, try with the next separator
#             if len(seps) > 1 and len(max(splited, key=len)) > max_len:
#                 splited = limit_len_text(splited, max_len, seps[1:])
#             return_text = return_text + splited
#         else:
#             return_text.append(t)

#         i = i + 1

#     return return_text

root_dir = Path(__file__).parent.parent.absolute()


# Get the command line arguments
parser2 = argparse.ArgumentParser()
parser2.add_argument('--epub', type=Path, default='data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub',
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
    args.out = root_dir / 'out' / sanitize(args.epub.stem).replace(' ', '_').lower()

# load epib
parsed = parser.from_file(str(args.epub))
text = parsed["content"]
if args.test:
    text = text[:1000]


# make output directory
out_dir = Path(args.out)
if out_dir.exists():
    if not args.force:
        print('Output folder already exists. Use -f to overwrite.')
        exit(1)
    else:
        for f in out_dir.glob('*'):
            f.unlink()
        out_dir.rmdir()
out_dir.mkdir()
print(f'Output folder: {out_dir}')

@dataclass
class Writer:
    out_dir: Path
    tts: TTS
    
    def __post_init__(self):
        self.m3u = open(self.out_dir / 'playlist.m3u', 'w')
        self.m3u.write('#EXTM3U\n')
        self.chapter = 1

    def write_chapter(self, waveforms):
        wav_f = out_dir / f'{self.chapter}.wav'
        self.tts.synthesizer.save_wav(wav=waveforms, path=wav_f)
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
print('use_cuda', use_cuda)
tts = TTS2(args.model, gpu=use_cuda, progress_bar=True)
writer = Writer(out_dir, tts)

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
        print('Skipping text without words or numbers', t)
        continue
    print('text', t)
    
    wav = tts.tts(text=t, language="en", speaker_wav=args.speaker)
    waveforms += wav
    
    if len(waveforms) > 10000000:  # ~20G
        wav_f = out_dir / f'{i}.wav'
        writer.write_chapter(waveforms, wav_f)
        waveforms = []
        
if len(waveforms):  
    writer.write_chapter(waveforms)
writer.close()
