
# for lightgbm
df["feature"].fillna(-999,inplace=True)
# for NN
df["feature"].fillna(0,inplace=True)
# other a kind of FE
df["feature"].fillna(df.feature.mean(),inplace=True)
df["feature"].fillna(df.feature.max(),inplace=True)
df["feature"].fillna(df.feature.min(),inplace=True)
