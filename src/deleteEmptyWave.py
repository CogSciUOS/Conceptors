import sys
import os


def delEmpty(path):
    for root, dirs, files in os.walk(path, topdown=False):

        for d in dirs:
            direc = os.path.join(root, d)
            delEmpty(direc)
        
        for name in files:
            f = os.path.join(root, name)
            if os.path.getsize(f) < 1500:
                print(f)
                os.remove(f)

path = os.path.abspath('D:/Data/Projects/StudyProject/syll')
delEmpty(path)