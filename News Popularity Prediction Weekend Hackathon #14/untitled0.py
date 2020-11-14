# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:50:53 2020

@author: dineshy86
"""

import pandas as pd
df = pd.DataFrame()

9/21/1989
19640430
6/27/1980
5/11/1987
Mar 12 1951
2 aug 2015

dates = ['9/21/1989','19640430','6/27/1980','5/11/1987','Mar 12 1951','2 aug 2015']
columns= [] 
for i in range(5):
    columns.append('date_type_{}'.format(i))

dates = pd.DataFrame()



import re
from datetime import datetime

with open("in.txt","r") as fi, open("out.txt","w") as fo:
    for line in fi:
        line = line.strip()
        dateObj = None
        if re.match(r"^\d{8}$", line):
            dateObj = datetime.strptime(line,'%Y%m%d')
        elif re.match(r"^\d{1,2}/", line):
            dateObj = datetime.strptime(line,'%m/%d/%Y')
        elif re.match(r"^[a-z]{3}", line, re.IGNORECASE):
            dateObj = datetime.strptime(line,'%b %d %Y')
        elif re.match(r"^\d{1,2} [a-z]{3}", line, re.IGNORECASE):
            dateObj = datetime.strptime(line,'%d %b %Y')
        fo.write(dateObj.strftime('%m-%d-%Y') + "\n")
        

datetime.strptime('2 aug 2015','%d %b %Y')


result = re.match("^[a-z]{3}", '6/27/1980')

if result:
  print("Search successful.")
else:
  print("Search unsuccessful.")	






for date in dates:
    dateObj = None
    if re.match("^\d{8}$", line):
        dateObj = datetime.strptime(date,'%Y%m%d')
    elif re.match("^\d{1,2}/", line):
        dateObj = datetime.strptime(date,'%m/%d/%Y')
    elif re.match("^[a-z]{3}", line, re.IGNORECASE):
        dateObj = datetime.strptime(date,'%b %d %Y')
    elif re.match("^\d{1,2} [a-z]{3}", line, re.IGNORECASE):
        dateObj = datetime.strptime(date,'%d %b %Y')
    print(dateObj)
    
    

    fo.write(dateObj.strftime('%m-%d-%Y') + "\n")
    













