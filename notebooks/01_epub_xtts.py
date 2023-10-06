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



def limit_text_len(text, max_len=200, seps=['.', ',', '\n', ' ']) -> list:
    # If the some element of text is too long, divide it in smaller parts using the separators
    return_text = []
    i = 0
    for t in text:
        if len(t) > max_len:
            splited = t.split(seps[0])

            # Restore the separators
            for j in range(0, len(splited)-1):
                splited[j] = splited[j] + seps[0]

            # If the split is not good enough, try with the next separator
            if len(seps) > 1 and len(max(splited, key=len)) > max_len:
                splited = limit_text_len(splited, max_len, seps[1:])
            return_text = return_text + splited
        else:
            return_text.append(t)

        i = i + 1

    return return_text

root_dir = Path(__file__).parent.parent.absolute()


# Get the command line arguments
parser2 = argparse.ArgumentParser()
parser2.add_argument('--epub', type=Path, default='data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub',
                    help='PDF file to read')
parser2.add_argument('-o', '--out', type=Path, default=None, help='Output folder')
parser2.add_argument('-f', '--force', action='store_true', default=False, help='Overwrite')
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
    args.out = root_dir / 'out' / sanitize(args.epub.stem)

# load epib
parsed = parser.from_file(str(args.epub))
text = parsed["content"]

# generate speech by cloning a voice using default settings
# FIXME:
text = """Then imagine them when they are putting on airs; when they make those haughty gestures, or when they get angry and upbraid people with such a superior air.” (Meditations, IX.9)', 'In reality, here and in several other passages, Marcus is simply deploying the standard Stoic technique of adopting a broader, more neutral perspective, forcing himself to redescribe things in a more objective, less emotional way.', 'Why?', 'So that he can better deal with people and events that would otherwise be upsetting precisely because we look at them too closely, or in a manner that is too emotionally involved.', 'Another reason to apply this strategy is explained in clear by Seneca:', '“It is no less ridiculous to be shocked by these things than it is to complain because you get splashed in the baths, or shoved around in a public place, or that you get dirty in muddy places. What happens in life is exactly like what happens in the baths, in a crowd or on a muddy road. … Life is not made for delicate souls.” (Letters to Lucilius, 107.2)', 'None of the above means that we should adopt a quietist attitude and just let life happen to us.', 'It only means that we should strive to tackle life’s problems with reason, rather than being overwhelmed by irrational emotional attachments, and that we should realize that if we decide to walk on mud we are going, inevitably, to get dirty.', """ # TEST

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
# create a playlist
m3u = open(out_dir / 'playlist.m3u', 'w')
m3u.write('#EXTM3U\n')
last_top_chapter = None
last_chapter = None


# write metadata to dir
from json_tricks import dump, dumps, load, loads, strip_comments
f_metadata = out_dir / 'metadata.json'
with open(f_metadata, 'w') as fo:
    dump(dict(
        epub_metadata=parsed['metadata'],
        args=args.__dict__,
        
    ), fo, indent=4)

# load model
use_cuda = torch.cuda.is_available()
print('use_cuda', use_cuda)
tts = TTS(args.model, gpu=use_cuda, progress_bar=True)

# split text
text = limit_text_len([text], args.limit)
# text = [text]
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
    print(t)
    wav_f = out_dir / f'{i}.wav'
    tts.tts_to_file(text=t,
                    file_path=wav_f,
                    speaker_wav=args.speaker,
                    language="en")
    waveforms.append(wav_f)
    
    
    # wav = tts.tts(text=t, language="en", speaker_wav=args.speaker)
    # tts.synthesizer.save_wav(wav=wav, path=wav_f)
    # pdb.set_trace()

    # Add the file to the playlist
    if last_top_chapter != out_dir:
        m3u.write(f'#EXTGRP:{out_dir}\n')
        last_top_chapter = out_dir
    if last_chapter != i:
        m3u.write(f'#EXTINF:-1,{i}\n')
        last_chapter = i
    m3u.write(f'{out_dir}[{i}]{i}.wav\n')
m3u.close()
