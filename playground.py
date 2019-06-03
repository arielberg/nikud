#!/usr/bin/python3
import os
import re
import pandas as pd
import random


def prepareLine():
    a = 1

df = pd.read_csv('data/googleML.csv', header=None)
print(df.head())



'''

path = "/www/nikod/raw/bible"

files = []
content =  ""
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            with open(path+"/"+file, "r") as f:
                content = content+(f.read())

chars = ' אבגדהוזחטיכלמנסעפצקרשתךםןףץ'
f = open("nikud.txt","r")

nikods = ""
line = f.readline()
while line:
    line = f.readline()
    line = line.strip()
    linep = line.split(':')
    if(len(linep)==2):
        nikods = nikods+linep[1]
f.close()

content = re.sub("[^"+chars+nikods+"]", "", content)

print(content)
with open("tanach.txt", "w") as f:
    f.write(content)
'''
