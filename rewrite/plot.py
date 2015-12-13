__author__ = 'admin'
import pandas as pd
import matplotlib.pyplot as plt
import os
os.chdir("D:/fall 2015/ecs 289")
seg = pd.read_csv("segment.scale.csv")
del seg[u'Unnamed: 0']
del seg['iter']
grouped = pd.rolling_mean(seg[seg.rate > 0.005].groupby(['rate', 'num']).mean(), 4)
grouped.plot()
plt.show()

dna = pd.read_csv("dna.scale.csv")
del dna[u'Unnamed: 0']
del dna['iter']
grouped = pd.rolling_mean(dna[dna.rate > 0.005].groupby(['rate', 'num']).mean(), 4)
grouped.plot()
plt.show()


