from sklearn.base import BaseEstimator, TransformerMixin

class HighMissingDropper(BaseEstimator, TransformerMixin):
    """
    Drops columns with a high percentage of missing values.
    
    Parameters
    ----------
    threshold : float, default=0.4
        The proportion of missing values above which a column will be dropped.
        Must be between 0 and 1.
    
    Attributes
    ----------
    columns_to_drop_ : list of str
        Names of columns identified for removal during fitting.
    """
    
    def __init__(self, threshold=0.4):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Learn which columns to drop based on missing value threshold.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset to analyze for missing values.
        y : array-like, optional
            Target values (ignored).
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        missing_counts = X.isnull().sum()
        missing_threshold = X.shape[0] * self.threshold
        self.columns_to_drop_ = missing_counts[missing_counts > missing_threshold].index.tolist()
        return self
    
    def transform(self, X):
        """
        Remove columns with high missing value proportions.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset to transform.
            
        Returns
        -------
        X_transformed : pandas.DataFrame
            Dataset with high-missing columns removed.
        """
        return X.drop(columns=self.columns_to_drop_)