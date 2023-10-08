the 01 xtts script sounds good, but keep having assert errors. At first I thought it was because of large framements to I used the best splitter: a recusrive splitter by token

but even with chunks far below the limit I get that assert

it's werird because in cpu mode it seems to come from an mebdding far above the embedding limt

like my longest chunk is 117
but the assert is at 600 why?

last thing I'll try is a very small chunk


# links

- https://github.com/suno-ai/bark/blob/main/notebooks/long_form_generation.ipynb
- https://gist.github.com/endes0/0967d7c5bb1877559c4ae84be05e036c"


Note I am now using my fork of TTS https://github.com/wassname/TTS


# 2023-10-08 11:21:08

OK I switched to tortise, it has a similar quality to TTS, but I find it way less buggy and simpler to work with. I did have to play with the requirements a bit to get it working, but it's documents in requirements.txt
