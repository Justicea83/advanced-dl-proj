import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class MusicTracksDataset(Dataset):
    """Music Tracks dataset."""

    def __init__(self, csv_file, train=True, test_size=0.3, random_state=42, transform=None, mean=None, std=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            train (bool): If True, use the training set. If False, use the test set.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility of the splits.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        df = self.cleaned_data(csv_file)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

        self.tracks_frame = train_df if train else test_df
        self.genre_to_int = {genre: i for i, genre in enumerate(df['track_genre'].unique())}
        self.mean = mean
        self.std = std

    @classmethod
    def cleaned_data(cls, csv_file):
        df = pd.read_csv(csv_file)
        # Ensure 'track_genre' is not dropped in this step
        df = df.drop(['track_id', 'artists', 'album_name', 'Unnamed: 0', 'track_name'], axis=1)
        return df

    @classmethod
    def compute_mean_std(cls, csv_file):
        df = cls.cleaned_data(csv_file).drop(['track_genre'], axis=1)
        mean = df.mean().values
        std = df.std().values
        return mean, std

    def get_input_size(self):
        return self.tracks_frame.shape[1] - 1

    def get_output_size(self):
        return len(self.tracks_frame['track_genre'].unique())

    def __len__(self):
        return len(self.tracks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Ensure features does not include 'track_genre'
        features = self.tracks_frame.drop(['track_genre'], axis=1).iloc[idx]
        features = torch.tensor(features.values.astype('float32'))
        if self.mean is not None and self.std is not None:
            features = (features - torch.tensor(self.mean, dtype=torch.float32)) \
                       / torch.tensor(self.std, dtype=torch.float32)

        genre_string = self.tracks_frame.iloc[idx]['track_genre']
        label = self.genre_to_int[genre_string]
        label = torch.tensor(label, dtype=torch.long)

        sample = {'features': features, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
