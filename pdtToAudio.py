import torchaudio
import PyPDF2
import subprocess
import os
import torch
import re
import argparse
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface


# helpers
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


def force_limit_len(text, max_len=200) -> list:
    # Splice the string elements of text to make them fit in max_len
    return_text = []
    i = 0
    for t in text:
        if len(t) > max_len:
            splited = [t[i:i+max_len] for i in range(0, len(t), max_len)]
            return_text = return_text + splited
        else:
            return_text.append(t)
        i = i + 1

    return return_text


def generate_index(pdfobj, outlines, top='', recur=True) -> dict:
    result = {}
    last = None
    for bookmark in outlines:
        if hasattr(bookmark, 'title') and bookmark.title is not None:
            title = re.sub(' +', ' ', bookmark.title.replace('\n',
                           '').replace('/', '').replace('\\', '').strip())
            result[top + title] = pdfobj.get_destination_page_number(bookmark)
            last = title
        elif type(bookmark) is list and recur:
            result.update(generate_index(
                pdfobj, bookmark, top + last + '->', True))

    return result


def search_page_in_index(index, page):
    # search for the value with the minimum difference
    min_diff = 1000000
    min_key = None
    for key, value in index.items():
        if value == page:
            return key
        elif value < page:
            diff = page - value
            if diff < min_diff:
                min_diff = diff
                min_key = key

    return min_key if min_key != None else 'No chapter'


# Get the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pdf', type=str, default='example.pdf',
                    help='PDF file to read')
parser.add_argument('--out', type=str, default='out', help='Output folder')
parser.add_argument('--page', type=int, default=1,
                    help='Page to start reading')
parser.add_argument('--limit', type=int, default=350,
                    help='Maximum number of characters to synthesize at once')
parser.add_argument('--model', type=str, default='facebook/fastspeech2-en-ljspeech',
                    help='fairseq model to use from HuggingFace Hub')
parser.add_argument('--vocoder', type=str, default='hifigan',
                    help='Vocoder to use from the model')
parser.add_argument('--speaker', type=int, default=0,
                    help='Speaker to use from the model')
args = parser.parse_args()


# Intialize TTS and Vocoder
models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
    args.model,
    arg_overrides={"vocoder": args.vocoder, "fp16": False}
)
model = models[0]
TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
generator = task.build_generator([model], cfg)

# check if the model is in GPU
use_cuda = torch.cuda.is_available()
if use_cuda:
    print("Using GPU")
    model.cuda()

# read the PDF file
pdfReader = PyPDF2.PdfFileReader(args.pdf)

# number of pages in the PDF file
print(pdfReader.numPages)

# get the chapter names
index = generate_index(pdfReader, pdfReader.outlines)
top_index = generate_index(pdfReader, pdfReader.outlines, recur=False)

# show the index
for key, value in index.items():
    print(key, value)

# create the output folders
os.mkdir(args.out)
os.mkdir(args.out + '/No chapter')
for key in top_index.keys():
    os.mkdir(args.out + '/' + key)

# create a playlist
m3u = open(args.out + '/playlist.m3u', 'w')
m3u.write('#EXTM3U\n')
last_top_chapter = None
last_chapter = None

# iterate through the pages
i = args.page - 1
for page in pdfReader.pages:
    i = i + 1
    # extract the text from the page using pdftotext command
    text = subprocess.check_output(
        ['pdftotext', '-f', str(i), '-l', str(i), '-layout', args.pdf, '-']).decode('utf-8')

    # Remove \x0c characters
    text = text.replace('\x0c', '')

    # Remove duplicated spaces
    text = re.sub(' +', ' ', text)
    text = text.strip()

    # Check if the page is empty
    if text == None or text == '':
        continue

    # Divide the text
    text = limit_text_len([text], args.limit)
    #text = force_limit_len(text, 200)

    waveforms = []
    for t in text:
        t = t.replace('\n', ' ').strip()
        # Skip empty text
        if t == None or t == '':
            continue

        # check if contains words or numbers
        if not re.search('[a-zA-Z0-9]', t):
            continue
        print(t)

        # Running the TTS
        sample = TTSHubInterface.get_model_input(
            task, t, verbose=False)
        if use_cuda:
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].cuda()
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].cuda()
            sample["speaker"] = sample["speaker"].cuda(
            ) if sample["speaker"] is not None else None
        wav, rate = TTSHubInterface.get_prediction(
            task, model, generator, sample)

        waveforms.append(wav)

    # Concatenate the waveforms
    if len(waveforms) == 0:
        continue
    waveforms = torch.cat(waveforms).repeat(1, 1)

    # Get the chapter name
    chapter = search_page_in_index(index, i+1)
    out_dir = (search_page_in_index(top_index, i+1) + '/')

    # Save the waverform
    torchaudio.save(os.path.join(args.out, out_dir,
                    '[' + str(i) + '] ' + chapter + '.wav'), waveforms.cpu(), task.sr)

    # Add the file to the playlist
    if last_top_chapter != out_dir:
        m3u.write('#EXTGRP:' + out_dir[:-1] + '\n')
        last_top_chapter = out_dir
    if last_chapter != chapter:
        m3u.write('#EXTINF:-1,' + chapter + '\n')
        last_chapter = chapter
    m3u.write(out_dir + '[' + str(i) + '] ' + chapter + '.wav\n')

m3u.close()
