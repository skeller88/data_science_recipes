import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm


def plot_normal_distribution(data):
    sns.distplot(data, fit=norm)

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(data)
    print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    # Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
                loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    # Get the QQ-plot
    fig = plt.figure()
    res = stats.probplot(data, plot=plt)
    plt.show()
