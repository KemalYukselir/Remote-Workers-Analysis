import pandas as pd

##################
### Import CSV ###
##################

# pd.set_option('display.max_columns', None)
df = pd.read_csv('data/2020_rws.csv', encoding='ISO-8859-1')

print(df.info())