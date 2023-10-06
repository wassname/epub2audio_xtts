import os
import sys


# Read all the files in the current directory and subdirectories
def read_files(path):
    files = []
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            files.append(os.path.join(root, f))
    return files

# Convert the files to ogg
def convert(files):
    for f in files:
        if f.endswith('.wav'):
            print(f)
            os.system('ffmpeg -i "' + f + '" -acodec libvorbis -aq 4 "' + f[:-4] + '".ogg')
            os.remove(f)

# Add metadata to the ogg files
def add_metadata(files):
    for f in files:
        if f.endswith('.ogg'):
            print(f)
            splited = f.split('/')[-1][:-4].split('->')
            os.system('vorbiscomment -a -t TITLE="' + splited[-1] + '" "' + f + '"')
            os.system('vorbiscomment -a -t ALBUM="' + (" - ".join(splited[1:-2]) if len(splited) > 2 else " - ".join(splited[0:-1])) + '" "' + f + '"')



path = sys.argv[1]
files = read_files(path)
convert(files)
files = read_files(path)
add_metadata(files)
