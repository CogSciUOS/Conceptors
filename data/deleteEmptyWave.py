
import os


def delEmpty(path):
    for root, dirs, files in os.walk(path, topdown=False):

        for d in dirs:
            direc = os.path.join(root, d)
            delEmpty(direc)

        for name in files:
            f = os.path.join(root, name)

            if os.path.getsize(f) < 1500 and name != '.gitignore':
                os.remove(f)

path = os.path.abspath('birddb/syll')
delEmpty(path)
