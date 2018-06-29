plt.figure(figsize=(12,8))
sns.distplot(train_df["feature"].values, bins=100, kde=False)
plt.xlabel('target', fontsize=12)
plt.title("target / feature Histogram", fontsize=14)
plt.show()

# other method
cnt_srs = train_df['feature'].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Feature in Train'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="Feature")
