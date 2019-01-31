from sklearn.decomposition import PCA


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        del X_df_new['PO53']
        del X_df_new['PO50']
        del X_df_new['PO71']
        del X_df_new['PO74']
        del X_df_new['SD5']
        del X_df_new['SD1']
        X_df = X_df.iloc[:,0:60]
        sklearn_pca = PCA(n_components=9)
        X_sklearn = sklearn_pca.fit_transform(X_df_new)
        X_df_new = X_sklearn
        return X_df_new