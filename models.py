"""
Building predictive model on regression principles.
@AugustSemrau
"""


from data_loader import dataLoader

# SciKit-Learn
from sklearn.linear_model import LinearRegression



class Models:
    """Class containing all classification models"""
    # Init
    def __init__(self):
        self.X, self.y = dataLoader(test=False, optimize_set=False, return_all=False)
        self.y = self.y.values.ravel()

    # Linear Regression Model
    def build_model_lr(self):
        model = LinearRegression()
        model.fit(self.X, self.y)
        return model

