"""
Building predictive model on regression principles.
@AugustSemrau
"""


from data_loader import dataLoader

# SciKit-Learn
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

"""
Class containing all classification models
"""
class Models:

    # Init
    def __init__(self):
        self.X, self.y = dataLoader(test=False, optimize_set=False, return_all=False)
        self.y = self.y.values.ravel()

    # Baseline model which simply guesses the mean value of house prices
    def build_model_baseline(self):
        model = DummyRegressor(strategy="mean")
        model.fit(self.X, self.y)
        return model

    # Linear Regression Model
    def build_model_lr(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model




