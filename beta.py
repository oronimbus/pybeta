"""Feature different approaches to calculating security betas. 

References:
-----------
    Blitz, David, Laurens Swinkels, Kristina Ūsaitė, and Pim van Vliet. 'Shrinking Beta'. 
        SSRN Scholarly Paper. Rochester, NY, 10 February 2022.
    Blume, Marshall E. 'Betas and Their Regression Tendencies'. The Journal of Finance 30,
        no. 3 (1975): 785-95.
    Frazzini, Andrea, and Lasse Heje Pedersen. 'Betting against Beta'. Journal of Financial
        Economics 111, no. 1 (1 January 2014): 1-25.
    Hollstein, Fabian, Marcel Prokopczuk, and Chardin Wese Simen. 'Estimating Beta: Forecast
        Adjustments and the Impact of Stock Characteristics for a Broad Cross-Section'.
        SSRN Scholarly Paper. Rochester, NY, 17 August 2018.
    Scholes, Myron, and Joseph Williams. 'Estimating Betas from Nonsynchronous Data'.
        Journal of Financial Economics 5, no. 3 (1 December 1977): 309-27.
    Welch, Ivo. 'Simply Better Market Betas'. SSRN Scholarly Paper. Rochester, NY, 13 June 2021. 
    Vasicek, Oldrich A. 'A Note on Using Cross-Sectional Information in Bayesian Estimation of
        Security Betas'. The Journal of Finance 28, no. 5 (December 1973): 1233-39.
"""
import numpy as np


class Beta:
    def __init__(self, exog, endog):
        self.exog = np.atleast_2d(exog).T
        self.endog = np.atleast_2d(endog).T
        self.n_obs = exog.shape[0]
        self.exog_mat = np.hstack([np.ones((self.n_obs, 1)), self.exog])

    def _weighted_ols(
        self, X: np.array, y: np.array, w: np.array = None, demean: bool = False
        ) -> np.array:
        """Helper class to calculate beta using WLS.

        Args:
            X (np.array): exogeneous variable (e.g. SPY)
            y (np.array): endogeneous variable (e.g. AAPL)
            w (np.array, optional): vector of weights. Defaults to None.
            demean (bool, optional): subtract mean from ``X``. Defaults to False.

        Returns:
            np.array: vector of betas
        """        
        if demean:
            X -= np.mean(X, axis=0)
        
        if w is None:
            w = np.ones(X.shape[0])
        w = np.diag(w)
        return np.linalg.inv(X.T @ w @ X) @ X.T @ w @ y
    
    def ols(self, adjusted: bool = False) -> float:
        """Classic beta calculation using OLS.
        
        Beta can be shrunk towards unity using Merril Lynch approach. This is the same 
        as Blume (1975).

        Args:
            adjusted (bool, optional): shrink towards unity. Defaults to False.

        Returns:
            float: beta
        """
        beta = np.ravel(self._weighted_ols(self.exog_mat, self.endog))[-1]
        if adjusted:
            return 0.67 * beta + 0.33
        return beta
        
    
    def vasicek(self, beta_prior: float = 1, se_prior: float = 0.5) -> float:
        """Bayesian estimation of beta using Vasicek (1973).

        Args:
            beta_prior (float, optional): _description_. Defaults to 1.
            se_prior (float, optional): _description_. Defaults to 0.5.

        Returns:
            float: Vasicek beta
        """
        beta = self._weighted_ols(self.exog_mat, self.endog)
        endog_hat = self.exog_mat @ beta
        s_yy = np.sum(np.square(self.endog - endog_hat)) / (self.n_obs - 2)
        s_xx = np.sum(np.square(self.exog - np.mean(self.exog)))
        std_error = np.sqrt(s_yy / s_xx)
        
        # Bayesian estimation of marginal posterior  
        num = beta_prior / np.square(se_prior) + beta[1] / np.square(std_error)
        den = 1 / np.square(se_prior) + 1 / np.square(std_error)
        return np.ravel(num / den)[0]
        
    
    def welch(self, delta: float = 3, rho: float = 0) -> float:
        """Slope winsorized beta using Welch (2021).

        A decay factor ``rho`` can be chosen such that more relevance is given
        to more recent observation. The paper uses ``2/256`` as an exponential
        decay factor.
    
        Args:
            delta (float, optional): winsorisation parameter. Defaults to 3.
            rho (float, optional): decay factor. Defaults to 0.

        Returns:
            float: Welch beta
        """
        bm_min, bm_max = (1 - delta) * self.exog, (1 + delta) * self.exog
        lower, upper = np.minimum(bm_min, bm_max), np.maximum(bm_min, bm_max)
        endog_wins = np.atleast_2d(np.clip(self.endog, lower, upper))
        weights = np.exp(-rho * np.arange(self.n_obs)[::-1])
        beta = self._weighted_ols(self.exog_mat, endog_wins, w=weights)
        return np.ravel(beta)[1]
        
    def robeco(
        self, corr_target: float, vol_target: float, gamma: float = 0.5, phi: float = 0.2
        ) -> float:
        """Beta shrinkage using Blitz et al. (2022).

        The Robeco beta is calculated using the correlation of asset to market returns and the
        ratio of their volatilties separately. Both are shrunk towards a cross-sectional mean.
        
        Args:
            corr_target (float): cross sectional mean of correlation
            vol_target (float): cross sectional mean of volatility ratio
            gamma (float, optional): correlation shrinkage factor. Defaults to 0.5.
            phi (float, optional): volatility ratio shrinkage factor. Defaults to 0.2.

        Returns:
            float: _description_
        """
        corr = np.corrcoef(self.exog.T, self.endog.T)[0,1]
        corr_shrink = (1 - gamma) * corr + gamma * corr_target
        vol_ratio = np.std(self.endog) / np.std(self.exog)
        vol_shrink = (1 - phi) * vol_ratio + phi * vol_target
        beta = corr_shrink / vol_shrink
        return beta

    def scholes_williams(self) -> float:
        """Calculate shrunk beta using Scholes & Williams (1977)."""
        beta_lag = np.ravel(self._weighted_ols(self.exog_mat[1:,:], self.endog[:-1,:]))[-1]
        beta_lead = np.ravel(self._weighted_ols(self.exog_mat[:-1,:], self.endog[1:,:]))[-1]
        beta = np.ravel(self._weighted_ols(self.exog_mat, self.endog))[-1]
        auto_corr = np.corrcoef(self.exog[1:,:].T, self.exog[:-1,:].T)[0,1]
        
        beta = (beta_lag + beta + beta_lead) / (1 + 2 * auto_corr)
        return beta
        
    def bayesian_model_avg(self):
        pass
        