cnt_srs = train_df['category'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='category distribution of ...',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="cate/..")
