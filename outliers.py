import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyasl

#read the dataframe; 'All Subject Areas' is the name of the worksheet in the template
df = pd.read_excel('nameOfYourFile.xlsx', sheetname = 'All Subject Areas')

#make a dataframe based on topic--'Subject Area' is the column name in the template
dfSubject = df[df['Subject Area'] == 'A subject area']

#make an array of the FWCIs--'FWCI' is the column name in the template
dfSubjectFwci = dfSubject[['FWCI']].as_matrix()

#make an array of the publications--'Scholarly Output" is the column name in the template
dfSubjectPubs = dfSubject[['Scholarly Output']].as_matrix()

#use pyasl to identify outliers in FWCI
r = pyasl.generalizedESD(df_bdP, 10, 0.05, fullOutput = True)
print('Number of outliers: ', r[0])
print('indices of outliers: ', r[1])

#get the median of the publications
np.median(dfSubjectPubs)