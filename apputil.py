import pandas as pd
import numpy as np

class GroupEstimate:
    def __init__(self, estimate="mean"):
        """
        Fit a group-level estimator that computes mean or median
        of a target variable y for groups defined by categorical features X.

        Parameters:
        estimate: str, either "mean" or "median"
            The type of group-level estimate to compute.

        Raises:
            ValueError: if estimate is not "mean" or "median"

        Returns:
        self
        """
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be either 'mean' or 'median'")
        self.estimate = estimate
        self.group_estimates_ = None
        self.group_cols_ = None

    def fit(self, X, y):
        """
        Fit the estimator by grouping categorical features in X
        and computing mean/median of y for each group.
        
        Parameters:
            X: pd.DataFrame - categorical features
            y: pd.Series or np.array - continuous target variable to estimate
        
        Raises:
            ValueError: if lengths of X and y do not match

        Returns:
        self
        """
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Ensure y is a Series
        y = pd.Series(y, name="target")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Combine into one DataFrame
        df = pd.concat([X, y], axis=1)

        # Save column names for later
        self.group_cols_ = list(X.columns)

        # Group and aggregate
        if self.estimate == "mean":
            self.group_estimates_ = df.groupby(self.group_cols_)["target"].mean()
        else:
            self.group_estimates_ = df.groupby(self.group_cols_)["target"].median()

        return self

    def predict(self, X_):
        """
        Predict estimates for new observations.

        Parameters:
            X_: pd.DataFrame - new categorical features for prediction

        Raises:
            RuntimeError: if fit has not been called before predict
        
        Returns:
            np.array - predicted group-level estimates
        """
        if self.group_estimates_ is None:
            raise RuntimeError("You must call .fit() before .predict()")

        # Ensure DataFrame
        if not isinstance(X_, pd.DataFrame):
            X_ = pd.DataFrame(X_, columns=self.group_cols_)

        # Join with stored group estimates
        merged = X_.merge(
            self.group_estimates_.reset_index(),
            on=self.group_cols_,
            how="left"
        )

        # Count missing groups
        missing_count = merged["target"].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} observations had unseen groups.")

        return merged["target"].values