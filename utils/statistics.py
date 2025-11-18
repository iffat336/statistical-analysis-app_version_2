import statsmodels.api as sm
import pandas as pd

def run_linear_regression(df, x_cols, y_col):
    """
    df: pandas DataFrame
    x_cols: list of predictor column names
    y_col: dependent variable
    """
    X = df[x_cols]
    X = sm.add_constant(X)  # add intercept
    y = df[y_col]
    model = sm.OLS(y, X).fit()
    return model

def run_anova(df, dv, factor):
    """
    Simple one-way ANOVA
    dv: dependent variable
    factor: categorical grouping variable
    """
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    formula = f"{dv} ~ C({factor})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return anova_table
