import numpy as np
from sklearn.linear_model import LinearRegression

class FederatedLinearRegression(LinearRegression):
	def fit(self, X, y, sample_weight=None):
		self.num_samples_ = len(X)
		super().fit(X, y, sample_weight)

	@staticmethod
	def federated_average(flr_model_list):
		flr_averaged = FederatedLinearRegression()

		# Zero initialize federated average model
		flr_averaged.num_samples_ = 0
		flr_averaged.coef_ = np.zeros(len(flr_model_list[0].coef_))
		flr_averaged.intercept_ = 0.0

		for flr in flr_model_list:
			# Average coefficients
			flr_averaged_coef = ((flr.coef_ * flr.num_samples_) + (flr_averaged.coef_ * flr_averaged.num_samples_)) / (flr.num_samples_ + flr_averaged.num_samples_)

			# Average intercepts
			flr_averaged_intercept = ((flr.intercept_ * flr.num_samples_) + (flr_averaged.intercept_ * flr_averaged.num_samples_)) / (flr.num_samples_ + flr_averaged.num_samples_)

			# Update averaged model with new averaged coefficients and intercept
			flr_averaged.num_samples_ += flr.num_samples_
			flr_averaged.coef_ = flr_averaged_coef
			flr_averaged.intercept_ = flr_averaged_intercept

		return flr_averaged