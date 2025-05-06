import pandas as pd

##################
### Import CSV 
##################

df = pd.read_csv('data/2020_rws.csv', encoding='ISO-8859-1')

#####################
### Adjust col names
#####################

name_sort_df = df.copy()
result = []
# print(type(name_sort_df.columns))
for col in name_sort_df.columns:
    if len(col) < 121:
        result.append(col[0:120])
    else:
        result.append(col[-121:])

name_sort_df.columns = result
name_sort_df.info()



