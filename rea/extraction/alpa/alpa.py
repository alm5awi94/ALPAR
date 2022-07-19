import logging
from typing import Tuple

import numpy as np
import tensorflow.keras.models as models
from rpy2 import robjects

from data.data import Data
from evaluation.evaluate_rules.fidelity import fidelity
from extraction.alpa.alpa_c5 import c5_r_predict, get_c5_model

logger = logging.getLogger(__name__)


class Alpa:
    """The ALPA algorithm for pedagogical rule extraction from ANNs."""

    def __init__(self, data: Data):
        """
        Set the attributes as in the original ALPA paper.

        The paper can be found at:
        `doi <https://doi.org/10.1109/TNNLS.2015.2389037>`_

        :param data: The `Data` module instance of the pipeline.

        """
        # numbers from ALPA paper
        self._rho_interval = 0.05
        self._valleypoints_per = 0.25
        # number of training samples
        self.Nt = len(data.x_train)
        # number of valley points
        self.Nv = round(self._valleypoints_per * self.Nt)
        self.data = data

    def alpa(self,
             model: models.Model,
             seed: int = 42) -> Tuple[robjects.vectors.ListVector, dict]:
        """
        Run the ALPA algorithm on the given model and dataset

        :param model: The trained model which to extract the rules from
        :param seed: Seed for the RNG

        :return: ``Set()`` of rules and some metrics

        """
        np.random.seed(seed)
        # number of training samples
        # Nt = len(data.x_train)
        # probabilities for classification
        oracle_train_y_prob = model.predict(
            np.reshape(self.data.x_train, self.data.original_shape), verbose=1)
        # predicted classes
        oracle_train_y_pred = np.argmax(oracle_train_y_prob, axis=1)
        # number of valley points
        # Nv = round(_valleypoints_per * self.Nt)
        valleypts, valleypt_clss = self.generate_valleypoints(
            self.data.x_train, oracle_train_y_prob)

        # index of the nearest point with different class
        # vector of size Nv with indices
        nearest = self.get_nearest(valleypt_clss, valleypts)
        logger.debug(f"Found nearest points to {self.Nv} valleypoints.")
        # generate some rulesets and determine one with the highest fidelity
        best_whitebox = None
        best_rho: float = self._rho_interval
        max_fidelity = float("-inf")

        # 250% magic number from ALPA paper
        if self.Nt < 10000:
            # [0.1, 0.2, ..., 2.5]
            rhos = (i * self._rho_interval for i in range(2, 51, 2))
        else:
            # avoid runtime explosion
            rhos = (0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05)

        for rho in rhos:
            # number of random samples generated between valley points
            # generate at least 1 point to avoid errors
            Nr = max(round(rho * self.Nt), 1)
            artificial = self.generate_points(valleypts, Nr, nearest)
            # label generated samples and build augmented dataset
            artificial_pred = np.argmax(model.predict(np.reshape(
                artificial, self.data.original_shape)), axis=1)
            gen_samples = np.concatenate([self.data.x_train, artificial])
            gen_labels = self.data.inverse_transform_classes(
                np.concatenate([oracle_train_y_pred, artificial_pred]))
            # train whitebox
            whitebox = get_c5_model(x=gen_samples, y=gen_labels, seed=seed)
            whitebox_pred = self.data.inverse_transform_classes(
                c5_r_predict(whitebox, gen_samples))
            # evaluate and update best ruleset
            fid = fidelity(gen_labels, whitebox_pred)
            logger.debug(
                f"Generated {Nr} points (rho={rho:.2}) - fid: {fid:.5}")
            if fid > max_fidelity:
                logger.debug(
                    f"Found better ruleset for {Nr} "
                    f"(rho={rho:.2}) new samples "
                    f"({fid:.5} > {max_fidelity:.5})")
                max_fidelity = fid
                best_rho = rho
                best_whitebox = whitebox

        return best_whitebox, {"best_rho": best_rho, "max_fid": max_fidelity}

    def generate_valleypoints(self, x_train: np.ndarray,
                              oracle_train: np.ndarray,
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get `Nv` points where neural network has lower confidence.

        :param x_train: The training data of the neural network.
        :param oracle_train: The prediction of the neural network for `x_train`

        :return: `Nv` indices in `x_train` and corresponding predictions.

        """
        # Classification:
        #   1 take confidence/probability for class
        #   2 subtract values for other classes -> more confidence in other
        #     classes negatively impacts score
        # then sort by score and take first Nv

        # Regression would be difference to expected value
        #  scores = np.abs(oracle_train - y_train)

        preds = np.argmax(oracle_train, 1)
        if (preds == preds[0]).all():
            raise RuntimeError("Your network seems to predict the same class "
                               "for every training instance. Check whether "
                               "it overfits. ALPA can not work in this case, "
                               "as there is no decision boundary. "
                               f"Predictions: {preds}")

        scores = np.max(oracle_train, axis=1)  # optimized for softmax
        # if last layer is not softmax use:
        #  2 * np.max(oracle_train, axis=1) - np.sum(oracle_train, axis=1)

        # get Nv indices for low score points
        indices = np.argsort(scores)[
            :self.Nv]  # if Regression: indices[-Nv:] ?
        valleypoints = x_train[indices]  # fancy indexing
        valleypoint_classes = np.argmax(oracle_train[indices], axis=1)
        return valleypoints, valleypoint_classes

    def get_nearest(self, predicted_classes: np.ndarray,
                    points: np.ndarray) -> np.ndarray:
        """
        Get the next nearest neighbor with different class per valleypoint.

        :param predicted_classes: The neural network prediction per point.
        :param points: The points to calculate nearest neighbour for.

        :return: Vector with indices of nearest neighbor in `points`.

        """
        if (predicted_classes == predicted_classes[0]).all():
            raise RuntimeError("All valleypoints have the same class. "
                               "Cannot calculate nearest neighbor "
                               "with different class for "
                               f"Predicted classes: {predicted_classes}")

        self.Nv = len(points)
        neighbors = np.zeros(self.Nv, int)  # nearest neighbor with diff class
        # pairwise comparison of classes gives Boolean matrix of shape Nv x Nv
        mask = predicted_classes[:, None] != predicted_classes

        for vp, m, i in zip(points, mask, range(self.Nv)):
            # loop per valleypoint to avoid quadratic memory usage
            # only calculate for non equal classes
            to_calc = np.nonzero(m)[0]  # get indices in points
            # calculate squared euclidean distance by broadcasting vp
            distances = np.sum((vp[None, :] - points[to_calc]) ** 2, axis=1)
            neighbors[i] = to_calc[
                np.argmin(distances)]  # get back correct index in points

        return neighbors  # Return the vector with nearest neighbors

    def generate_points(self, valleypoints: np.ndarray,
                        Nr: int, neighbor_indices: np.ndarray) -> np.ndarray:
        """
        Get new points in the decision boundary (area between `valleypoints`).

        :param valleypoints: The points with low confidence.
        :param Nr: Number of points to generate.
        :param neighbor_indices: Vector with indices of nearest neighbor with
         different class in `valleypoints`.

        :return: `Nr` generated points,
         each point is a linear combination between nearest `valleypoints`

        """
        # sample Nr random valleypoints
        indices_1 = np.random.randint(self.Nv, size=Nr)
        vp_1 = valleypoints[indices_1]
        # get nearest neighbor index to create pairs
        indices_2 = neighbor_indices[indices_1]
        vp_2 = valleypoints[indices_2]
        # linear combination of the pairs creates new points
        thetas = np.random.rand(Nr)  # Nr numbers in [0,1)
        # apply each theta to all attribute values per point (broadcast)
        generated = thetas[:, None] * vp_1 + (1 - thetas)[:, None] * vp_2
        return generated
