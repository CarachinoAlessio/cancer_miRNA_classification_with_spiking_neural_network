import pandas as pd
import os


class FeatureSelection():
    def __init__(
            self,
            feature_path: str,
            n_superclasses: int
        ) -> None:
        self.features = self.__load_features(feature_path=feature_path)
        self.n_superclasses = n_superclasses

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
    
    def reduce_dimensionality(self, data, method='fisher', train=True) -> pd.DataFrame:
        # TODO
        pass