# Beta Shrinkage Methods

This repo gives alternative ways to look at beta for security analysis, strategy development and hedging. The traditional definition of beta in finance quantifies by how much an asset moves if its benchmark changes. The beta is therefore the non-diversifiable, systemic market risk and is given by $\beta_{i,t}=cov(r_{i,t},r_{m,t}) / \sigma^2_{m,t}$ at any time $t$. It might be more intuitive to think of it as the correlation of stock $i$ to market $m$ returns multiplied by a volatility ratio: $\beta_{i_t}=\rho\times \frac{\sigma_{i,t}}{\sigma_{m,t}}$. This also establishes an explicit link between Pearson's correlation coefficient and the normal equation of the OLS estimator. Typically, the returns are expressed as *excess returns* (net of risk free returns) which is being omitted here.

Multiple papers have been published on how best to estimate beta, some offering simple improvements to the OLS version and others more complex, iterative procedures (see references). A great summary can be found in Hollstein et al. (2018) and Welch (2021).

Do note that simply having a better estimator for security beta does *not* guarantee you to make money (if that wasn't obvious already)! However, it might help removing some of the noise when dealing with financial data. Similarly, having a more complex model does not automatically result in a better estimation of beta per se.

Finally, this is by no means a *study* of beta estimation but merely a demonstration of various implementations. I am using a single, arbitrarily chosen estimation horizon (1 month) across all examples. One ought to compare long term estimation (e.g. 3 years) as well as shorter, intraday horizons. 

## Installation
To install you can either create your own dist and install it by running 
```
python setup.py bdist_wheel && cd dist && pip install pybeta-0.1.0-py3-none-any.whl
```
or just download and `pip install` the wheel directly from the righthand side.


## References
- Blitz, David, Laurens Swinkels, Kristina Ūsaitė, and Pim van Vliet. 'Shrinking Beta'.  SSRN Scholarly Paper. Rochester, NY, 10 February 2022.
- Blume, Marshall E. 'Betas and Their Regression Tendencies'. The Journal of Finance 30, no. 3 (1975): 785-95.
- Frazzini, Andrea, and Lasse Heje Pedersen. 'Betting against Beta'. Journal of Financial Economics 111, no. 1 (1 January 2014): 1-25.
- Hollstein, Fabian, Marcel Prokopczuk, and Chardin Wese Simen. 'Estimating Beta: Forecast Adjustments and the Impact of Stock Characteristics for a Broad Cross-Section'. SSRN Scholarly Paper. Rochester, NY, 17 August 2018.
- Scholes, Myron, and Joseph Williams. 'Estimating Betas from Nonsynchronous Data'. Journal of Financial Economics 5, no. 3 (1 December 1977): 309-27.
- Welch, Ivo. 'Simply Better Market Betas'. SSRN Scholarly Paper. Rochester, NY, 13 June 2021. 
- Vasicek, Oldrich A. 'A Note on Using Cross-Sectional Information in Bayesian Estimation of Security Betas'. The Journal of Finance 28, no. 5 (December 1973): 1233-39.
