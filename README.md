# Trying to make an audiobook from an epub file 2023-10-07 09:53:29

After trying a few AI TTS models I like
- xTTS
- Tortise
- Bark.

Here I'm using the TTS library... which is changing rapidly. So this code might be out of date fast. However you might find parts interesting, e.g. epub reading, splitting.





sse https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_1_use-pretrained-TTS.ipynb
https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c

# setup

```sh
conda create -n tts python=3.10       
conda activate tts       
pip install ipykernel TTS               

python notebooks/01_epub_xtts.py  --epub "data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub" --out short_guide_inner_citadel
python toogg.py # WIP not working yet
```

see also:



UPTO: see [research notes](./research_notes.md)
