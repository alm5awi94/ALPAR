import json
import logging
import os
import pickle
from typing import Dict, List, Tuple, Union

import category_encoders as ce
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import class_weight

logger = logging.getLogger(__name__)


class Data:
    """Module to handle the data acquisition and preprocessing."""

    def __init__(
            self,
            input_path: str,
            output_path: str,
            label_col: Union[int, str],
            dataset_name: str,
            original_shape: List[int] = None,
            test_size: float = 0.3,
            cat_conv_method: str = "woe",
            categorical_columns: Union[List[str], List[int]] = None,
            seed: int = 42, scale_data: bool = False,
    ):
        """
        Create a new instance of the data module.

        :param input_path: Filepath to the dataset in csv format
        :param output_path: Path to the folder to fill with output
        :param label_col: index or name of the column containing the labels
        :param dataset_name: friendly name to give to the dataset
        :param test_size: percentage of the data used for testing
        :param categorical_columns: List of categorical columns to be
            converted in the dataset
        :param cat_conv_method: Method for categorical conversion
        :param seed: random seed for data splitting
        :param scale_data: flag to MinMaxScale the input data

        """
        logger.debug("Created new Data instance.")
        # private fields
        self._input_path = input_path
        self._output_path = output_path
        self._label_col = label_col
        self._dataset_name = dataset_name
        self._test_size = test_size
        self._cat_conv_method = cat_conv_method
        self._seed = seed
        self._class_encoder = LabelEncoder()
        self._data_scaler = MinMaxScaler()
        # public fields
        self.x_train: np.ndarray = np.zeros(1)
        self.x_test: np.ndarray = np.zeros(1)
        self.y_train: np.ndarray = np.zeros(1)
        self.y_test: np.ndarray = np.zeros(1)
        self.class_weights: Dict[int, float] = {}
        if categorical_columns is None:
            categorical_columns = []
        self.categorical_columns: Union[
            List[str], List[int]] = categorical_columns
        self.categorical_woe_encoding = {}
        self.original_shape = original_shape
        self.feature_names: List[str] = []
        self.scale_data = scale_data

    @property
    def class_names(self):
        return self._class_encoder.classes_

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def num_features(self):
        return len(self.feature_names)

    def _one_hot_encode_classes(self, labels: np.ndarray, n_classes: int) -> \
            Tuple[np.ndarray, np.ndarray]:
        # Transform from e.g. string to numbers
        labels = self._class_encoder.fit_transform(labels)
        labels_categorical = to_categorical(labels, num_classes=n_classes)
        return labels_categorical, labels

    def run(self) -> "Data":
        """
        Apply the default data processing.

        :return: this instance of data updated with the processed data.

        """
        logger.debug("Running default data processing.")
        logger.debug("Attempting to load data.")
        path, filename = os.path.split(self._input_path)
        ext = filename.split(".")[1]
        if ext == "csv":
            dataset = pd.read_csv(self._input_path)
        elif ext == "hdf5" or ext == "h5":
            dataset = pd.read_hdf(self._input_path)
        else:
            raise ValueError(f"Unsupported filetype of {self._input_path}."
                             f"Supported filetypes are: csv, hdf")
        x, y, class_names, self.feature_names = self._split_labels(
            dataset, self._label_col)
        num_classes = len(class_names)
        _, _, y_train_vanilla, _ = self._split_test(x, y, self._seed,
                                                    self._test_size)
        # one hot encoding after class_weights determination
        self.class_weights = self._get_class_weights(y_train_vanilla)
        del y_train_vanilla
        y, y_labels_encoded = self._one_hot_encode_classes(y, num_classes)
        if len(self.categorical_columns) > 0:
            x = self.convert_categorical(dataset, x, y_labels_encoded)
            logger.debug(f"New attr names: {self.feature_names}")
        if self.scale_data:
            x = self._data_scaler.fit_transform(x)
        self.x_train, self.x_test, self.y_train, self.y_test = \
            self._split_test(x, y, self._seed, self._test_size)

        if self.original_shape is None:
            self.original_shape = [-1, self.num_features]
        else:
            self.original_shape.insert(0, -1)
        self._write()
        return self

    def convert_categorical(self, dataset: pd.DataFrame, x: np.ndarray,
                            y_labels_encoded: np.ndarray):
        if self._cat_conv_method == "woe":
            # woe needs indices instead of column names
            if type(self.categorical_columns[0]) == str:
                categorical_columns = [dataset.columns.get_loc(c)
                                       for c in self.categorical_columns
                                       if c in dataset]
            else:
                categorical_columns = self.categorical_columns

            x, self.categorical_woe_encoding, self.feature_names = \
                self._convert_categorical_woe(x, y_labels_encoded,
                                              categorical_columns,
                                              self.class_names,
                                              self.feature_names)
        else:
            # scikit-learn functions need column string names
            if type(self.categorical_columns[0]) == int:
                categorical_columns = [self.feature_names[int(i)]
                                       for i in self.categorical_columns]
            else:
                categorical_columns = self.categorical_columns
            x, self.feature_names = \
                self._conv_cat_scikit(x, self.feature_names,
                                      categorical_columns,
                                      self._cat_conv_method, y_labels_encoded)
        return x

    def _write(self):
        """Writes the processed data and metadata to the output path."""
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        np.save(os.path.join(self._output_path, "x_train.npy"), self.x_train)
        np.save(os.path.join(self._output_path, "x_test.npy"), self.x_test)
        np.save(os.path.join(self._output_path, "y_train.npy"), self.y_train)
        np.save(os.path.join(self._output_path, "y_test.npy"), self.y_test)
        with open(os.path.join(self._output_path, "encoder.pickle"),
                  'wb') as f:
            pickle.dump(self._class_encoder, f)
        if self.scale_data:
            with open(os.path.join(self._output_path, "scaler.pickle"),
                      'wb') as f:
                pickle.dump(self._data_scaler, f)
        metadata = {
            "input_path": self._input_path,
            "output_path": self._output_path,
            "label_col": self._label_col,
            "dataset_name": self._dataset_name,
            "test_size": self._test_size,
            "seed": self._seed,
            "class_weights": self.class_weights,
            "original_shape": self.original_shape,
            "feature_names": self.feature_names,
            "categorical_columns": self.categorical_columns,
            "cat_conv_method": self._cat_conv_method,
            "scale_data": self.scale_data
        }
        with open(os.path.join(self._output_path, "metadata.json"), "w+") as f:
            json.dump(metadata, f)

    def inverse_transform_classes(self, encoded_classes: np.ndarray):
        """
        Transform integer encoded classes back to label.

        :param encoded_classes: Integer encoded classes

        :return: Labels of the classes.

        """
        return self._class_encoder.inverse_transform(encoded_classes)

    def inverse_transform_scaling(self, encoded_value: float, attr_index):
        """
        Transform ``MinMaxScaled`` value back to original.

        :param encoded_value: Float encoded value
        :param attr_index: Index of the feature the value belongs too

        :return: Original, pre-scaled value.

        """
        # TODO make this similiar to inverse_transform_classes?
        # would move sample gen and selecting on the inverse outside function

        # create dummy sample to inverse transform
        sample = np.zeros((1, self.num_features))
        sample[0, attr_index] = encoded_value
        # and pick just the value in the column we actually want
        return self._data_scaler.inverse_transform(sample)[0, attr_index]

    @classmethod
    def _conv_cat_scikit(cls, x: np.ndarray, feature_names: List[str],
                         categorical_columns: List[str],
                         cat_conv_key: str,
                         y: np.ndarray) -> Tuple[np.ndarray, list[str]]:
        """
        Convert the categorical columns in the given data points to numerical
        using a scikit learn method from ``Data.cat_encoder_methods``.

        :param x: The input dataset to convert columns from
        :param categorical_columns: The list of column names for x
        :param categorical_columns: The list of categorical column names
        :param cat_conv_key: Name encoding of scikit-learn method

        :return: The converted data of ``x`` and the new feature_names

        """
        logger.debug(f"Converting to categorical with {cat_conv_key} method.")
        encoder_params = dict(cols=categorical_columns, return_df=False)
        if cat_conv_key == "onehot":
            encoder_params['use_cat_names'] = True
        encoder = cls.cat_encoder_methods[cat_conv_key](**encoder_params)
        df_x = pd.DataFrame(data=x, columns=feature_names)
        transformed = encoder.fit_transform(df_x, y)
        return transformed, encoder.feature_names

    @staticmethod
    def read(path: str) -> "Data":
        """
        Reconstructs a data instance from a previously saved one.

        :param path: The path to the output folder.

        :return: A new data instance.

        """
        params: dict
        with open(os.path.join(path, "metadata.json"), "r") as f:
            params = json.load(f)
        d = Data(
            params["input_path"], params["output_path"],
            params["label_col"], params["dataset_name"],
            params["original_shape"], params["test_size"],
            params["cat_conv_method"], params["categorical_columns"],
            params["seed"], params["scale_data"]
        )
        d._class_weights = params["class_weights"]
        d.feature_names = params["feature_names"]
        d.x_train = np.load(os.path.join(path, "x_train.npy"))
        d.x_test = np.load(os.path.join(path, "x_test.npy"))
        d.y_test = np.load(os.path.join(path, "y_test.npy"))
        d.y_train = np.load(os.path.join(path, "y_train.npy"))
        with open(os.path.join(d._output_path, "encoder.pickle"), 'rb') as f:
            d._class_encoder = pickle.load(f)
        if d.scale_data:
            with open(os.path.join(
                    d._output_path, "scaler.pickle"), 'rb') as f:
                d._data_scaler = pickle.load(f)
        return d

    @staticmethod
    def _get_class_weights(y_train: np.ndarray) -> Dict[int, float]:
        """
        Calculate the weights for each class to determine the balancing of the
        dataset.

        :param y_train: The model labels

        :return: The weights for each class.

        """
        return dict(enumerate(class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train)))

    @staticmethod
    def _split_test(
            x: np.ndarray, y: np.ndarray, seed: int, test_size: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the dataset into model and testing datasets using
        scikit-learns train_test_split

        :param x: The attributes
        :param y: The labels
        :param seed: The seed to use for the rng
        :param test_size: The proportion of entries used for testing

        :return: Tuple of model, testing data in the form ``(x_train, x_test,
            y_train, y_test)``

        """
        if test_size != 0.0:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=seed, test_size=test_size, shuffle=True)
        else:
            x_train, x_test, y_train, y_test = x, x, y, y
        return x_train, x_test, y_train, y_test

    @staticmethod
    def _split_labels(dataset: pd.DataFrame, label_col: Union[int, str]) \
            -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Split the given dataset into attributes and labels and transform it to
        an array

        :param dataset: The dataset to split
        :param label_col: The index or name of the column containing the labels

        :return: Tuple of attributes, labels of type np.ndarray, class
            and feature names

        """

        # determine index of label column if necessary
        label_col_name = label_col
        if type(label_col_name) == int:
            label_col_name = dataset.columns[label_col]

        # split labels and transform to numpy arrays
        x: np.ndarray = dataset.drop(label_col_name, axis=1).to_numpy()
        y: np.ndarray = dataset[label_col_name].to_numpy()

        class_names = dataset[label_col_name].drop_duplicates().to_list()
        feature_names = dataset.columns.to_list()
        feature_names.remove(label_col_name)
        return x, y, class_names, feature_names

    @staticmethod
    def _convert_categorical_woe(x: np.ndarray,
                                 y: np.ndarray,
                                 categorical_columns: List[int],
                                 class_names: List[str],
                                 feature_names: List[str]) \
            -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]], List[str]]:
        """
        Convert the categorical columns in the given data points to numerical
        using WoE Encoding.

        Since WoE encoding is normally based on binary decisions, we split
        an n-dimensional decision into n binary decisions and calculate a
        vector based encoding on those. See
        `here <https://www.listendata.com/2015/03/weight-of-evidence-woe-and-
        information.html>`_
        for more detail on WoE

        :param x: The input dataset to convert columns from
        :param y: The decisions for each point in x
        :param categorical_columns: The list of column indices of x
            that are to be converted
        :param class_names: The list of class names

        :return: Tuple of converted points, leaving the original columns
            intact and a Dictionary containing a Dictionary for each
            converted column, where the latter includes the encodings
            for each column value

        """
        logger.debug("Converting to categorical with woe method.")
        # save encoding for each feature
        categorical_encoded: dict[str, dict[str, np.ndarray]] = {}
        num_classes = len(class_names)
        # first count number of occurrences for specific output label
        _, num_events_sum = np.unique(y, return_counts=True)
        # and number of non-occurrences for specific output label
        # both are used for WoE calculation
        num_points = x.shape[0]
        num_non_events_sum = num_points - num_events_sum
        # Would require LabelEncoding the categorical columns to remove
        # the need for value_labels and idx and bincount for counts.
        # Using minlength in bincount allows for padding with non existent
        # attribute values to fix dimensionality issues. But has higher
        # memory requirements, since for value_counts the categorical
        # matrix needs to be unravelled and y needs to copied for each
        # num_events_sum bincount call (since we're going through y for
        # each categorical feature)
        for i in categorical_columns:
            for c in class_names:
                feature_names.append(f"{feature_names[i]}_{c}")
            # need both values and their occurrence amount
            value_labels, idx, value_counts = \
                np.unique(x[:, i], return_counts=True, return_inverse=True)
            # save amount of different values for convenience
            num_labels = value_labels.shape[0]

            # For WoE calculation we first need to save how often each
            # attribute value is mapped to each possible output class.
            # In order to do that for all attribute values all at once
            # we add an offset on the output vector, which depends on
            # the attribute value. Multiplying by num_classes is equivalent
            # to multiplying by the max value of y. This causes the number
            # of bins np.bincount counts to raise from num_classes to
            # num_classes*num_labels, therefore we have all the counts
            # we need at once. Finally we reshape to the matrix form
            # we want, where each row includes the counts for a specific
            # attribute value
            num_events = np.bincount(
                y + (num_classes * np.arange(num_labels))[idx],
                minlength=num_classes * num_labels) \
                .reshape(num_labels, -1)
            # The only bad thing about this is how this will take longer
            # when y gets larger (=larger dataset), since the addition
            # will take longer then
            # For comparison is a loop-based method below, which applies
            # np.bincount for each attribute value to a part of y which is
            # sliced for that specific attribute value
            # num_events = np.apply_along_axis(lambda k: np.bincount(
            # y[x[:,i]==k],minlength=num_classes),axis=1,arr=labels[:,None])
            # This would be independent of the size of y (to an extent, since
            # bincount still goes through y once overall, but that also
            # happens in the method above), but has python loop overhead

            # Comparing both methods on UCI (30000 data points) yielded
            # performance slowdown magnitude of apply_along_axis equal
            # to the number of repetitions of the loop. Meaning
            # apply_along_axis takes #num_labels times the time of
            # bincounting on the whole array with an offset in this
            # application. Therefore bincounting stands. Since this
            # doesn't conclude if this performance difference holds
            # over different dataset sizes the alternative still stands
            # in comments above

            # Then we also want to save how often each value is NOT mapped
            # to each output class
            num_non_events = value_counts[:, None] - num_events

            # here we adjust values where num_events or num_non_events
            # is 0, since that leaves math errors in the calculation
            # follows the handling of https://doi.org/10.15439/2015F90
            # minus the case where both are 0, this is handled
            # since the attribute value simply doesn't exist then
            # TODO can this be done in single steps instead of two per
            #  array?
            num_non_events = np.where(num_events == 0,
                                      num_non_events +
                                      num_non_events_sum / num_events_sum,
                                      num_non_events)
            num_events = np.where(num_events == 0, 1, num_events)
            num_events = np.where(num_non_events == 0,
                                  num_events +
                                  num_events_sum / num_non_events_sum,
                                  num_events)
            num_non_events = np.where(num_non_events == 0, 1,
                                      num_non_events)

            # then save WoE calculation
            # WoE = ln(event% / non-event%), including restructuring
            # of the fraction
            woe = np.log((num_events * num_non_events_sum) /
                         (num_events_sum * num_non_events))

            # save finished dict for category
            categorical_encoded[feature_names[i]] = \
                dict(zip(value_labels, woe))
            # Finally, we want to put the new columns. Here the index list
            # from before is helpful again, since the woe vectors can be
            # indexed the same way to easily encode the attribute values
            x = np.concatenate((x, woe[idx]), axis=1)
        x = np.delete(x, categorical_columns, axis=1)
        feature_names = [x for i, x in enumerate(feature_names)
                         if i not in categorical_columns]
        return x, categorical_encoded, feature_names

    cat_encoder_methods = {"onehot": ce.OneHotEncoder,
                           "helmert": ce.HelmertEncoder,
                           "leave_one_out": ce.LeaveOneOutEncoder,
                           "woe-scikit": ce.WOEEncoder,  # for testing
                           "ordinal": ce.OrdinalEncoder,
                           "woe": _convert_categorical_woe}
