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
for col in name_sort_df.columns:
    if len(col) < 121:
        result.append(col[0:120])
    else:
        result.append(col[-121:])

name_sort_df.columns = result
# name_sort_df.info()

#####################
### Check unique values
#####################

for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}: {df[col].unique()}\n")

#####################
### Check nulls
#####################

df_clean = name_sort_df.copy()
missing = df_clean.isnull().sum()

# Calculate % of missing values
missing_percent = (missing / len(df_clean)) * 100

# Filter and show only columns with missing data
missing_percent = missing_percent[missing_percent > 0]

# Format nicely
print(missing_percent.sort_values(ascending=False).round(2))



