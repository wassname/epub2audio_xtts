# Tasks

Development tasks for [mask](https://github.com/jacobdeichert/mask).

## run

~~~sh
python notebooks/01_epub_xtts.py  --epub "data/A Short Guide to the Inner Citadel - Massimo Pigliucci.epub" --out short_guide_inner_citadel
~~~


## freeze
> record pip and conda requirements

~~~bash
set -x -e
source activate tts
mkdir -p requirements
conda env export --no-builds --from-history > requirements/environment.min.yaml
conda env export > requirements/environment.max.yaml
python -m pip freeze > requirements/pip.conda.txt
cd requirements && conda-lock -f environment.max.yaml -p linux-64
~~~
