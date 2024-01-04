from typing import Any

from skfeature.function.similarity_based import fisher_score

from src.utils.data_loading_functions import load_data, split_data, normalize_data, class_balancing
from src.utils.utils import data_split_to_superclasses
import numpy as np
import pandas as pd
import os


class FeatureSelection:
    def __init__(
            self,
            feature_file_path: str,
            superclasses: list[list]
    ) -> None:
        self.features = None
        self.feature_file_name = feature_file_path
        self.superclasses = superclasses
        self.n_superclasses = len(self.superclasses)
        self.save_path = os.path.join('data', 'features')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def load_features(self) -> pd.DataFrame:
        '''
        Check if filepath exists and then loads the features

        Args:
            - feature_path (`str`): features filepath

        Returns:
            - `pd.DataFrame`
        '''
        assert os.path.isfile(os.path.join(self.save_path, self.feature_file_name))
        self.features = pd.read_csv(os.path.join(self.save_path, self.feature_file_name))
        return self.features

    def find_features_subset(self, data, method='fisher', n_features=300, export_to_csv=False):
        subsets, super_labels = data_split_to_superclasses(data, self.superclasses)
        selected_features = []
        if method == 'fisher':
            for i, superclass in enumerate(self.superclasses):
                data = np.asarray(subsets[i])
                labels = np.asarray(super_labels[i])
                # idx = fisher_score.fisher_score(data, labels, mode='rank')
                idx = fisher_score.fisher_score(data, labels, mode='index')
                num_features = n_features
                idx = idx[0:num_features]
                selected_features.append(idx)

            df_obj = {}

            for idx, feature in enumerate(selected_features):
                df_obj.setdefault(f'Super-class{idx}', feature)

            if export_to_csv:
                df = pd.DataFrame(df_obj)
                # f = open(os.path.join(self.save_path, 'selected_features.csv'))
                df.to_csv(os.path.join(self.save_path, self.feature_file_name))
                self.features = self.__load_features(feature_path=self.feature_file_name)
            return selected_features
        else:
            raise Exception(f'{method} not implemented... yet.')

    def reduce_dimensionality(self, data: np.ndarray[Any, np.dtype], inference_mode: bool = False,
                              data_category: str = 'train') -> tuple[list[Any], list[Any]]:
        '''
        :param data_category: 'train' | 'val'
        :param data: list containing samples (data + label)
        :param inference_mode: not used... yet
        :return: reduced data according to their superclass feature subset selection
        '''
        reduced_data = []
        data = np.asarray(data)
        subsets, labels = data_split_to_superclasses(data, self.superclasses)

        for i, superclass in enumerate(self.superclasses):
            subset = subsets[i]
            indici_features = self.features[f"Super-class{i}"].tolist()
            subset = np.asarray(subset).T

            subset = subset[indici_features]
            subset = subset.T
            reduced_data.append(subset)
            if not inference_mode:
                dataset = np.hstack((subset, np.asarray(labels[i]).reshape((len(labels[i]), 1))))

                filename = f'reduced_data_metalabel{i}_{data_category}.csv'
                file = os.path.join(self.save_path, filename)
                dataset = pd.DataFrame(dataset)
                dataset.to_csv(file)

        return reduced_data, labels

if __name__ == '__main__':
    # Load data
    label_path = os.path.join("data", "MLinApp_course_data", "tcga_mir_label.csv")
    data_path = os.path.join("data", "MLinApp_course_data", "tcga_mir_rpm.csv")
    miRNA_data, miRNA_labels, miRNA_tissues = load_data(data_path, label_path)

    # Adjust data
    miRNA_data, miRNA_labels, miRNA_tissues, labels, dictionary, lab = class_balancing(miRNA_data, miRNA_labels,
                                                                                    miRNA_tissues)
    # Z-Score normalization
    miRNA_data = normalize_data(miRNA_data)
    # Splitting the data
    train_data, test_data, train_label, test_label = split_data(miRNA_data, miRNA_labels)
    train_data, val_data, train_label, val_label = split_data(train_data, train_label)

    superclasses = [
        ['BRCA', 'KICH', 'KIRC', 'LUAD', 'LUSC', 'MESO', 'SARC', 'UCEC'],
        ['BLCA', 'CESC', 'HNSC', 'KIRP', 'PAAD', 'READ', 'STAD'],
        ['DLBC', 'LGG', 'PRAD', 'TGCT', 'THYM', 'UCS'],
        ['ACC', 'CHOL', 'LIHC'],
        ['ESCA', 'PCPG', 'SKCM', 'THCA', 'UVM']
    ]

    train_data = np.hstack((train_data, train_label.reshape((len(train_label), 1))))
    val_data = np.hstack((val_data, val_label.reshape((len(val_label), 1))))

    fs = FeatureSelection('filter_features.csv', superclasses)

    selected_features_idx = fs.find_features_subset(train_data, method='fisher', n_features=300, export_to_csv=True)

    reduced_train_data, train_labels = fs.reduce_dimensionality(train_data, data_category='train')
    reduced_val_data, val_labels = fs.reduce_dimensionality(val_data, data_category='val')
