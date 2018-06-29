from matplotlib_venn import venn2

plt.figure(figsize=(10,7))
venn2([set(train_df.feature.unique()), set(test_df.feature.unique())], set_labels = ('Train set', 'Test set') )
plt.title("Number of users in train and test", fontsize=15)
plt.show()
