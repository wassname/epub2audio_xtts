
se https://github.com/coqui-ai/TTS/blob/dev/notebooks/Tutorial_1_use-pretrained-TTS.ipynb
https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c

```sh
conda create -n tts python=3.10       
conda activate tts       
pip install ipykernel TTS               

python epubToAudio.py --epub "data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub" --out short_guide_inner_citadel
python toogg.py 
```
