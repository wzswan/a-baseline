total_df = pd.read_csv('total_df.csv')
total_df_all = pd.read_csv('total_df_all.csv')
print('total_df_all rows and col', total_df_all.shape)

def test_pca(data, create_plots=True):
    """Run PCA analysis, return embedding"""

    # Create a PCA object, specifying how many components we wish to keep
    pca = PCA(n_components=1000)

    # Run PCA on scaled numeric dataframe, and retrieve the projected data
    pca_trafo = pca.fit_transform(data)

    # The transformed data is in a numpy matrix. This may be inconvenient if we want to further
    # process the data, and have a more visual impression of what each column is etc. We therefore
    # put transformed/projected data into new dataframe, where we specify column names and index
    pca_df = pd.DataFrame(
        pca_trafo,
        index=total_df.index,
        columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
    )

    # Only construct plots if requested
    if create_plots:

        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )

        # Plot the cumulative explained variance
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # Show legend
        axes[0].legend(loc="best", frameon=True)

        

    return pca_df

# Run the PCA and get the embedded dimension
pca_df = test_pca(total_df)
pca_df_all = test_pca(total_df_all, create_plots=False)
pca_df.to_csv("pca_df.csv", index=False)
pca_df_all.to_csv("pca_df_all.csv", index=False)
