# Trying to make an audiobook from an epub file 2023-10-07 09:53:29

Here I'm using the Tortise model... which is changing rapidly. So this code might be out of date fast. However you might find parts interesting, e.g. epub reading, splitting, speed.

## Example output

"A short guide to the inner citadel" by Massimo Pigliucci. Read by Tortise trained on 12 seconds of Donald Robertson's voice.

<audio controls="controls">
  <source type="audio/ogg" src="data/a_short_guide_to_the_inner_citadel_-_massimo_pigliucci_donald_robertson.ogg"></source>
  [data/a_short_guide_to_the_inner_citadel_-_massimo_pigliucci_donald_robertson.ogg](./data/a_short_guide_to_the_inner_citadel_-_massimo_pigliucci_donald_robertson.ogg)
</audio>

## Entrypoint

see [notebooks/01_epub_tortise.ipynb](./notebooks/01_epub_tortise.ipynb)

## setup

```sh
conda create -n tts python=3.10       
conda activate tts       
pip install ipykernel TTS               

python notebooks/01_epub_xtts.py  --epub "data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub" --out short_guide_inner_citadel
python toogg.py # WIP not working yet
```

## links
- https://github.com/neonbjb/tortoise-tts/blob/main/tortoise_tts.ipynb
- use https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_1_use-pretrained-TTS.ipynb
- https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c


UPTO: see [research notes](./research_notes.md)
