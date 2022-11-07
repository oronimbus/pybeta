"""Feature different approaches to calculating security betas. 

References:
-----------
    Blitz, David, Laurens Swinkels, Kristina Ūsaitė, and Pim van Vliet. 'Shrinking Beta'. \
        SSRN Scholarly Paper. Rochester, NY, 10 February 2022.
    Blume, Marshall E. 'Betas and Their Regression Tendencies'. The Journal of Finance 30, \
        no. 3 (1975): 785-95.
    Frazzini, Andrea, and Lasse Heje Pedersen. 'Betting against Beta'. Journal of Financial \
        Economics 111, no. 1 (1 January 2014): 1-25.
    Hollstein, Fabian, Marcel Prokopczuk, and Chardin Wese Simen. 'Estimating Beta: Forecast \
        Adjustments and the Impact of Stock Characteristics for a Broad Cross-Section'. \
        SSRN Scholarly Paper. Rochester, NY, 17 August 2018.
    Scholes, Myron, and Joseph Williams. 'Estimating Betas from Nonsynchronous Data'. Journal \
        of Financial Economics 5, no. 3 (1 December 1977): 309-27.
    Stock, James H., and Mark W. Watson. 'Chapter 10 Forecasting with Many Predictors'. \
        In Handbook of Economic Forecasting, edited by G. Elliott, C. W. J. Granger, and A. \
        Timmermann, 1:515-54. Elsevier, 2006.
    Welch, Ivo. 'Simply Better Market Betas'. SSRN Scholarly Paper. Rochester, NY, 13 June 2021. 
    Vasicek, Oldrich A. 'A Note on Using Cross-Sectional Information in Bayesian Estimation of \
        Security Betas'. The Journal of Finance 28, no. 5 (December 1973): 1233-39.
"""
from itertools import chain, combinations
import numpy as np


def add_intercept(data: np.array) -> np.array:
    """Add column vector of ones to front of matrix."""
    return np.hstack([np.ones((data.shape[0], 1)), data])


class Beta:
    def __init__(self, exog: np.array, endog: np.array):
        """Initialise estimator class.

        Args:
            exog (np.array): benchmark or market return vector
            endog (np.array): asset or stock return vector
        """
        self.exog = np.atleast_2d(exog).T
        self.endog = np.atleast_2d(endog).T
        self.n_obs = exog.shape[0]
        self.exog_mat = add_intercept(self.exog)

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

    def ewma(self, half_life: float = 0.33) -> float:
        r"""Exponentially weighted moving average beta using WLS.

        Beta is calculated using WLS and the following weights vector:

        .. math::

            w = \frac{exp(-|t-\tau|h)}{\sum_{\tau=1}^{t-1}exp(-|t-\tau|h)}

        Where the half life is given by :math:`h=\frac{log(2)}{l}`.

        Args:
            half_life (bool, optional): half life of EWMA period. Defaults to 0.33.

        Returns:
            float: beta
        """
        h = np.log(2) / (self.n_obs * half_life)
        weights = np.exp(-np.abs(self.n_obs - np.arange(1, self.n_obs + 1)) * h)
        weights /= np.sum(weights)
        beta = self._weighted_ols(self.exog_mat, self.endog, w=weights)
        return np.ravel(beta)[1]

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

    def dimson(self, lags: int = 1) -> float:
        """Dimson (1979) beta estimator for infrequently traded assets.

        Args:
            lags (int, optional): number of market lags included. Defaults to 1.

        Returns:
            float: Dimson beta
        """
        X_trimmed, y_trimmed = self.exog[lags:], self.endog[lags:]
        X_lagged = [self.exog[lags - i : -i] for i in range(1, lags + 1)]
        X_mat = add_intercept(np.hstack([X_trimmed, *X_lagged]))
        beta = np.ravel(self._weighted_ols(X_mat, y_trimmed))[1:]
        idx = np.min([2, lags])
        return np.sum(beta[: idx + 1])

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
        The beta is the product of the correlation and the volatility ratio.

        Args:
            corr_target (float): cross sectional mean of correlation
            vol_target (float): cross sectional mean of volatility ratio
            gamma (float, optional): correlation shrinkage factor. Defaults to 0.5.
            phi (float, optional): volatility ratio shrinkage factor. Defaults to 0.2.

        Returns:
            float: Robeco beta
        """
        corr = np.corrcoef(self.exog.T, self.endog.T)[0, 1]
        corr_shrink = (1 - gamma) * corr + gamma * corr_target
        vol_ratio = np.std(self.endog) / np.std(self.exog)
        vol_shrink = (1 - phi) * vol_ratio + phi * vol_target
        beta = corr_shrink * vol_shrink
        return beta

    def scholes_williams(self, lag: int = 1) -> float:
        """Calculate shrunk beta using Scholes & Williams (1977).

        Args:
            lag (float, optional): offset used to calculate lead/lag betas. Defaults to 1.

        Returns:
            float: Scholes Williams beta
        """
        beta_lead = np.ravel(self._weighted_ols(self.exog_mat[lag:, :], self.endog[:-lag, :]))[-1]
        beta_lag = np.ravel(self._weighted_ols(self.exog_mat[:-lag, :], self.endog[lag:, :]))[-1]
        beta = np.ravel(self._weighted_ols(self.exog_mat, self.endog))[-1]
        auto_corr = np.corrcoef(self.exog[1:, :], self.exog[:-1, :], rowvar=False)[0, 1]

        beta = (beta_lag + beta + beta_lead) / (1 + 2 * auto_corr)
        return beta


class BetaForecastCombination:
    def __init__(self, exog: np.array, endog: np.array, window: int = 21):
        """Initialise exogeneous (X) and endogeneous (y) data."""
        self.exog = np.atleast_2d(exog).T
        self.endog = np.atleast_2d(endog).T
        self.window = window
        self.n_obs = self.endog.shape[0]
        self.weights = None

    def _generate_estimation_windows(self, data: np.array) -> list:
        """Generate list of ``t+k`` expanding windows."""
        return [data[: self.window + i] for i in range(data.shape[0] - self.window)]

    def _train_test_split(self) -> None:
        """Split data set into training and test periods given window cutoff."""
        cutoff = self.n_obs - self.window
        self.train_data = np.hstack([self.exog[:cutoff, :], self.endog[:cutoff, :]])
        self.test_data = np.hstack([self.exog[cutoff:, :], self.endog[cutoff:, :]])

    def _generate_betas(self, windows: list, **kwargs: dict) -> np.array:
        """Generate betas from ``Beta`` class.

        Args:
            windows (list): list of data windows

        Returns:
            np.array: ``Nxk`` matrix of betas.
        """
        # set up iterator
        beta_obj = [Beta(i[:, 0], i[:, 1]) for i in windows]
        c, v = kwargs.get("corr_target", 0.5), kwargs.get("vol_target", 2)

        # consumer iterator and cast into 2d numpy array
        ols = np.atleast_2d(list(map(lambda x: x.ols(), beta_obj))).T
        adj_ols = np.atleast_2d(list(map(lambda x: x.ols(True), beta_obj))).T
        vasicek = np.atleast_2d(list(map(lambda x: x.vasicek(), beta_obj))).T
        ewma = np.atleast_2d(list(map(lambda x: x.ewma(), beta_obj))).T
        dimson = np.atleast_2d(list(map(lambda x: x.dimson(), beta_obj))).T
        welch = np.atleast_2d(list(map(lambda x: x.welch(), beta_obj))).T
        aged_welch = np.atleast_2d(list(map(lambda x: x.welch(rho=2 / 256), beta_obj))).T
        robeco = np.atleast_2d(list(map(lambda x: x.robeco(c, v), beta_obj))).T
        schol_will = np.atleast_2d(list(map(lambda x: x.scholes_williams(), beta_obj))).T
        models = [ols, adj_ols, vasicek, ewma, dimson, welch, aged_welch, robeco, schol_will]

        return np.hstack(models)

    def fit(self) -> float:
        """Fit forecast combination model given window size.

        Returns:
            float: forecast combined beta.
        """
        # split training data from test data for parameter estimation
        self._train_test_split()

        # calculate beta using expanding window
        training_windows = self._generate_estimation_windows(self.train_data)
        betas = self._generate_betas(training_windows)

        # regress betas onto realised betas
        X_train = add_intercept(betas[:-1, :])
        self.weights = np.linalg.pinv(X_train) @ betas[1:, 0]

        # project weights onto test data
        betas_test = self._generate_betas([self.test_data])
        X_test = add_intercept(betas_test)
        return np.ravel(X_test @ self.weights)[0]


class BetaBMA(BetaForecastCombination):
    def __init__(self, exog: np.array, endog: np.array, window: int = 21):
        super().__init__(exog, endog, window)

    def _generate_beta_combinations(self, data: np.array) -> np.array:
        """Combine ``k`` columns in all possible ways.

        Args:
            data (_type_): array with ``k`` columns

        Returns:
            np.array: array of possible combinations of column indices
        """
        indices = list(range(data.shape[0]))
        combos = [np.array(list(combinations(indices, i))) for i in range(1, data.shape[0])]
        return np.array(list(chain(*combos)), dtype=object)

    def fit(self) -> float:
        """Fit Bayesian Model Averaging over specified window.

        Returns:
            float: beta from Bayesian Model Averaging.
        """
        # estimation windows for priors (train)
        self._train_test_split()
        training_windows = self._generate_estimation_windows(self.train_data)
        beta_train = self._generate_betas(training_windows)
        beta_test = add_intercept(self._generate_betas([self.test_data]))

        # get K beta combinations and set up model inputs
        combos = self._generate_beta_combinations(beta_train[0])
        g = 1 / min(len(training_windows), len(combos))
        a_g = g / (1 + g)

        # step 1: calculate SSR_r for restricted model
        # TODO: add variable number of lags into restricted model
        dof_r = 1
        beta_r = add_intercept(beta_train[:-dof_r, [0]])
        b_r_hat = np.linalg.pinv(beta_r) @ beta_train[dof_r:, [0]]
        ssr_r = np.sum(np.square(beta_train[dof_r:, [0]] - beta_r @ b_r_hat))

        # step 2: iterate over beta combinations, store as list of tuples
        models = []
        for combo in combos:
            # calculate SSR_u first using lagged realised betas
            beta_u = add_intercept(beta_train[:-1, combo])
            b_u_hat = np.linalg.pinv(beta_u) @ beta_train[1:, [0]]
            ssr_u = np.sum(np.square(beta_train[1:, [0]] - beta_u @ b_u_hat))

            # project combined beta onto series of realised betas
            weights = np.zeros((beta_train.shape[1] + 1, 1))
            weights[0, :] = b_u_hat[0, :]
            weights[combo + 1, :] = b_u_hat[1:, :]
            beta_k = np.ravel(beta_test @ weights)[0]
            models.append((combo.shape[0], beta_k, ssr_u))

        # step 3: calculate beta weights
        weights = np.zeros(len(models))
        for k, (p, _, ssr) in enumerate(models):
            weights[k] = np.power(a_g, p / 2) * np.power(1 + 1 / g * (ssr / ssr_r), -dof_r / 2)

        # step 4: combine weights with betas for final estimate
        beta_bma = np.sum(np.array(models)[:, 1] * weights / np.sum(weights))
        return beta_bma
