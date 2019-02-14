# Module import:
import pystan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

# Data simulation:
x = np.arange(1, 100, 5) # Returns evenly spaced values within a given interval.
y = 2.5 + .5 * x + np.random.randn(20) * 10 # Random values in a given shape.
N = len(x)
x = np.reshape(x,[20,1])
K = 1
 
# Plot the data:
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
# plt.savefig('Data.png')
plt.show()
sns.regplot(x=x, y=y)
"""In the simplest invocation, both functions draw a scatterplot of two variables, x and y,
and then fit the regression model y ~ x and plot the resulting regression line and a 95% confidence
interval for that regression"""
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
# plt.savefig('Reg_Data.png')
plt.show()


# STAN model:
regress_code = """
data {
 int<lower = 0> N; // the number of observations
 int<lower = 0> K; // the number of columns in the covariate matrix X
 real y[N]; // the vector of response variables
 matrix[N,K] X; // the covariate matrix X
}
parameters {
 real a; // intercept
 vector[K] b; // remaining regression parameters
 real<lower=0> sigma; // standard deviation of the error term
}
transformed parameters {
 vector[N] mu; // fitted values
 mu = a + X*b;
}
model {
 a ~ cauchy(0,10); //prior for the intercept following Gelman 2008
 for(i in 1:K)
  b[i] ~ cauchy(0,2.5); //prior for the slopes following Gelman 2008
 y ~ normal(mu, sigma);
}
"""

# Dictionary containing all data to be passed to STAN:
regress_data_dict = {'X': x, 'y': y, 'N': N, 'K': K}

# Fit the model:
n_iter = 1000
n_chains = 4
model = pystan.StanModel(model_code=regress_code)
fit = model.sampling(data=regress_data_dict, iter=n_iter, chains=n_chains)
"""There are n_chains chains, and each chain as n_iter runs. First half of the n_iter runs 
for each chain are warmup, the second half produce the posterior draws."""

# make a dataframe of parameter estimates for all chains
params = pd.DataFrame({'a': fit.extract()['a'], 'b': np.array([x[0] for x in fit.extract()['b']])})
medParam = params.median()

# Model summary:
print(fit)
 
# Show a traceplot of ALL parameters:
fit.traceplot()
# plt.savefig('Traceplot_all.png')
plt.show()
"""This plot show the traces of the parameters on the left for all chains and you want all chains to
converge to similar values (ie no divergence in the values). On the right side of the plot are the
posterior distributions of the parameters."""

# Show a traceplot for single parameter:
fit.plot(['a'])
# plt.savefig('Traceplot_a.png')
plt.show()

# Show a pairplot of the parameter:
sns.pairplot(params)
# plt.savefig('Pairplot.png')
plt.show()
"""A helpful summary plot is the pairwise correlation between the parameters, if each parameters
is adding additional independent information, the points should form a shapeless cloud. If you have 
strong correlation between several parameters, then you may consider dropping some as they do not add 
extra information."""

# Show a boxplot of the parameter:
sns.boxplot(params)
# plt.savefig('Boxplot.png')
plt.show()
"""Credible intervals are another summary for the different parameters in the models, the red bands 
in this graph show that the parameters have a probability of 0.8 to be within the bands. Which is an
easier summary than the classical frequentist confidence intervals which tells us: “If I repeat my
experiment many times, the values of my parameters will be within this interval 80% of the time”.

The box shows the quartiles of the dataset while the whiskers extend to show the rest of the
distribution, except for points that are determined to be “outliers” using a method that is a
function of the inter-quartile range."""


##### PREDICTION ####
# Make a prediction vector (the values of X for which you want to predict):
predX = np.arange(0, 100)
# Next, make a prediction function:
def stanPred(p):
	fitted = p[0] + p[1] * predX
	return pd.Series({'fitted': fitted})
# fitted values:
yhat = stanPred(medParam)
# Get the predicted values for all posterior draws. This is super convenient in pandas because
# it is possible to have a single column where each element is a list.
chainPreds = params.apply(stanPred, axis = 1) #this has n_iter*n_chains/2 rows and len(predX) columns

## PLOTTING
# create a random index for posterior draw sampling
idx = np.random.choice(int(n_iter*n_chains / 2.0)-1, 50)
# Plot the predicted values resulting from the sampled posterior draws.
# chainPreds.iloc[i, 0] gets predicted values from the ith sample
for i in range(len(idx)):
	plt.plot(predX, chainPreds.iloc[idx[i], 0], color='lightgrey')
# original data
plt.plot(x, y, 'ko')
# fitted values
plt.plot(predX, yhat['fitted'], 'k')
plt.xlabel('x')
plt.ylabel('y')
# plt.savefig('Plot1.png')
plt.show()

# Make a function that iterates over every predicted values in every posterior draw sample and returns
# the quantiles:
def quantileGet(q):
    # make a list to store the quantiles
    quants = []
    # for every predicted value
    for i in range(len(predX)):
        # make a vector to store the predictions from each chain
        val = []
        # next go down the rows and store the values
        for j in range(chainPreds.shape[0]):
            val.append(chainPreds['fitted'][j][i])
        # return the quantile for the predictions.
        quants.append(np.percentile(val, q))
    return quants
 
# NOTE THAT NUMPY DOES PERCENTILES, SO MULTIPLE QUANTILE BY 100
# 2.5% quantile
lower = quantileGet(2.5)
#97.5
upper = quantileGet(97.5)
 
# plot this
fig = plt.figure()
ax = fig.add_subplot(111)
# shade the credible interval
ax.fill_between(predX, lower, upper, facecolor = 'lightgrey', edgecolor = 'none')
# plot the data
ax.plot(x, y, 'ko')
# plot the fitted line
ax.plot(predX, yhat['fitted'], 'k')
# supplementals
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()
# plt.savefig('Plot2.png')
plt.show()