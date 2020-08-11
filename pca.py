"""
This script is used for intepreting the data visually via a Pricnipal Component Analysis.
@AugustSemrau
"""

from data_loader import dataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':

    # Load data
    X, y = dataLoader(test=False, optimize_set=False, return_all=False)

    # Standardizing the data
    Xs = pd.DataFrame(data=StandardScaler().fit_transform(X), columns=X.columns)
    # print(X)
    # print(Xs)

    # Setting up the PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(Xs)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
    pca_df = pd.concat([principal_df, y], axis=1)

    # Preparing a scatter plot using matplotlib.pyplot
    fig = plt.figure(figsize=(10, 10))
    pca_plot = fig.add_subplot(1, 1, 1)
    pca_plot.set_xlabel('Principal Component 1', fontsize=20)
    pca_plot.set_ylabel('Principal Component 2', fontsize=20)
    pca_plot.set_title('2 Component PCA', fontsize=25)

    # The 2D visualization will first be in regards to house prices being over/under mean house price
    mean_price = float(np.mean(y))
    print('Mean house price: ', mean_price)

    # Each row (observation) is plotted seperately as either red (under mean price) or green (over mean price)
    colors = ['r', 'g']
    for row in range(len(pca_df)):
        price = float(pca_df.loc[row, 'SalePrice'])
        if price < mean_price:
            color = 'r'
        else:
            color = 'g'
        pca_plot.scatter(pca_df.loc[row, 'principal_component_1'],
                         pca_df.loc[row, 'principal_component_2'],
                         c=color, s=20)

    # Add legend (this is a bad solution..)
    pca_plot.legend(['Over mean price', '', '', 'Under mean price'])
    # Show plot
    pca_plot.grid()
    plt.show()





