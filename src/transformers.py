import pandas as pd

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

class GroupedImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using median/mean within groups defined by categorical columns.
    Falls back to overall statistic for ungrouped missing values.
    
    Parameters
    ----------
    target_col : str
        Column name to impute missing values for
    group_cols : list of str, default=['Pclass', 'Sex']  
        Column names to group by for calculating imputation values
    strategy : str, default='median'
        Imputation strategy ('median' or 'mean')
    """
    
    def __init__(self, target_col, group_cols=[], strategy='median'):
        self.target_col = target_col
        self.group_cols = group_cols
        self.strategy = strategy
        
    def fit(self, X, y=None):
        """
        Learn group-based imputation values from training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Training data containing target column and grouping columns
        y : array-like, optional
            Target values (ignored)
            
        Returns
        -------
        self : object
        """
        # Calculate imputation values by groups
        self.group_values_ = {}
        
        if self.strategy == 'median':
            agg_func = 'median'
        elif self.strategy == 'mean':
            agg_func = 'mean'
        else:
            raise ValueError("strategy must be 'median' or 'mean'")
        
        # Group by specified columns and calculate statistic
        grouped = X.groupby(self.group_cols)[self.target_col].agg(agg_func)
        self.group_values_ = grouped.to_dict()
        
        # Calculate overall fallback value
        if self.strategy == 'median':
            self.overall_value_ = X[self.target_col].median()
        else:
            self.overall_value_ = X[self.target_col].mean()
            
        return self
        
    def transform(self, X):
        """
        Impute missing values using fitted group statistics.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Data to impute
            
        Returns
        -------
        X_imputed : pandas.DataFrame
            Data with imputed values
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ['group_values_', 'overall_value_'])
        
        X = X.copy()
        
        # Create a mask for missing values in target column
        missing_mask = X[self.target_col].isnull()
        
        if missing_mask.sum() > 0:  # Only proceed if there are missing values
            # Try to impute using group values
            for group_key, impute_value in self.group_values_.items():
                if pd.isna(impute_value):
                    continue
                    
                # Create mask for this specific group
                group_mask = missing_mask.copy()
                
                # Handle both single and multiple grouping columns
                if len(self.group_cols) == 1:
                    group_key = (group_key,)  # Convert to tuple for consistency
                
                for col, val in zip(self.group_cols, group_key):
                    group_mask = group_mask & (X[col] == val)
                
                # Impute for this group
                if group_mask.sum() > 0:
                    X.loc[group_mask, self.target_col] = impute_value
            
            # Fill any remaining missing values with overall statistic
            remaining_missing = X[self.target_col].isnull()
            if remaining_missing.sum() > 0:
                X.loc[remaining_missing, self.target_col] = self.overall_value_
        
        return X
    