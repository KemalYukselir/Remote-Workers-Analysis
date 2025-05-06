import pandas as pd

##################
### Import CSV 
##################

df = pd.read_csv('data/clash_royale_cards.csv')

#######################
### Check unique values 
#######################

for col in df.columns:
    print(f"{col}: {df[col].unique()}")

###########################
### Check and handle nulls
##########################

df_clean = df.copy()
df_clean.info()
print(df_clean.isnull().sum())

# Only one null -> Can be dropped
df_clean.dropna(inplace=True)
print(df_clean.isnull().sum())

############################
### Check summary statistics
############################

print(df_clean.describe())


