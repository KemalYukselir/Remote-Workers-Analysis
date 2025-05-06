import pandas as pd

##################
### Import CSV 
##################

df = pd.read_csv('data/data.csv', encoding='ISO-8859-1')

#####################
### Check unique values 
#####################

for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}: {df[col].unique()}\n")

#####################
### Check nulls
#####################

df_clean = df.copy()
missing = df_clean.isnull().sum()

# Calculate % of missing values
missing_percent = (missing / len(df_clean)) * 100

# Filter and show only columns with missing data
missing_percent = missing_percent[missing_percent > 0]

# Format nicely
print(missing_percent.sort_values(ascending=False).round(2))



