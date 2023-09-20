import numpy as np
import pandas as pd
import os


class FeatureSelection():
    def __init__(
            self,
            feature_path: str,
            superclasses: list[list]
        ) -> None:
        self.features = self.__load_features(feature_path=feature_path)
        self.superclasses = superclasses
        self.n_superclasses = len(self.superclasses)

    def __load_features(self, feature_path: str) -> pd.DataFrame:
        '''
        Check if filepath exists and then loads the features

        Args:
            - feature_path (`str`): features filepath

        Returns:
            - `pd.DataFrame`
        '''
        assert os.path.isfile(feature_path)
        return pd.read_csv(feature_path)
    
    def reduce_dimensionality(self, data: list[list], inference_mode=False) -> pd.DataFrame:

        result = []
        data = np.asarray(data)
        for i, superclass in enumerate(self.superclasses):
            indici = [j for j, d in enumerate(data) if d[-1] in superclass]
            subset = data[indici]

            indici_features = self.features[f"Super-class{i}"].tolist()
            subset = np.asarray(subset).T

            subset = subset[indici_features]
            result.append(subset.T)
            print('hell')


        return result
        pass

    def find_feature_subset(self, data, method='fisher', n_features=300):
        pass


data = [
            [1,2,4,5,'easp'],
            [3,4,4,5,'fppp'],
            [9,9,4,5,'ciao']
        ]

superclasses = [
            ['easp'],
            ['fppp', 'ciao']
        ]

output = [
    [[1,2,4,5,'easp']],
    [[3,4,4,5,'fppp'], [9,9,4,5,'ciao']]
]


fs = FeatureSelection('data/features/dummy_features.csv', superclasses)
output = fs.reduce_dimensionality(data)
print('stop')
