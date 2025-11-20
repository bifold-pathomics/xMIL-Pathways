import os

import torch
import pandas as pd


class MetadataHandler:

    @staticmethod
    def load_split_metadata(split_path, metadata_dirs, subsets, label_cols=None, modalities=None, merge_on='case_id'):
        """
        Loads and fuses metadata and split information, filtered by the given subsets.

        :param split_path: (str)
        :param metadata_dirs: (list<str>)
        :param subsets: (list<str>)
        :param label_cols: (list<str>)
        :param modalities: (list<str>)
        :param merge_on: (str)
        :return: (pandas.DataFrame)
        """
        label_cols = [] if label_cols is None else label_cols
        # Read split
        split = pd.read_csv(split_path)
        split['subset'] = split['subset'].astype(str)  # for compatibility with 'subsets' arg
        split = split[split['subset'].isin(subsets)][[merge_on, 'subset'] + label_cols]
        # Read and merge case metadata
        case_metadata = pd.DataFrame()
        for idx, metadata_dir in enumerate(metadata_dirs):
            case_met_ = pd.read_csv(os.path.join(metadata_dir, "case_metadata.csv"))
            case_met_.insert(0, 'source_id', idx)
            case_metadata = pd.concat([case_metadata, case_met_], axis=0, ignore_index=True)
        split = split.merge(case_metadata, how='inner', on=merge_on, suffixes=(None, '_ctrl'))
        # Sanity check the given data sources
        if 'source_id_ctrl' in split:
            if split['source_id'].equals(split['source_id_ctrl']):
                split = split.drop('source_id_ctrl', axis=1)
            else:
                raise ValueError("Given data sources do not match the data sources of the split")
        # Read and merge modality-specific metadata
        if modalities is not None:
            for modality in modalities:
                modality_metadata = pd.DataFrame()
                for idx, metadata_dir in enumerate(metadata_dirs):
                    mod_met_ = pd.read_csv(os.path.join(metadata_dir, f"{modality}_metadata.csv"))
                    modality_metadata = pd.concat([modality_metadata, mod_met_], axis=0, ignore_index=True)
                split = split.merge(modality_metadata, how='inner', on=merge_on, suffixes=(None, '_ctrl'))
        return split
