import os
import json
from functools import partial

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_handler import MetadataHandler, SlideDataHandler


def _binary_annotation_label_fn(patch_annot, target_class='1', other_thr=0.0, target_thr=0.0,
                                include_unlabeled_samples=False):
    target_frac = json.loads(patch_annot).get(target_class)
    default_label = -1 if include_unlabeled_samples else None
    if target_frac is None:
        return default_label
    elif target_frac > target_thr:
        return 1
    elif target_frac <= other_thr:
        return 0
    else:
        return default_label


class PatchFeatureDataset(Dataset):

    def __init__(self, split_path, metadata_dirs, subsets, patches_dirs, features_dirs, label_cols,
                 thresholds=(0.0, 0.0), patch_filters=None, drop_duplicates='sample', sampling=None,
                 include_unlabeled_samples=False):
        super(PatchFeatureDataset, self).__init__()
        if len(label_cols) != 1:
            raise ValueError(f"Multiple labels not implemented for PatchFeatureDataset. Given: {label_cols}")
        # Save args
        self.features_dirs = features_dirs
        self.label_col = label_cols[0]
        self.sampling = sampling
        # Load metadata, slide data, and match them
        print(f"Loading dataset for subsets: {subsets}")
        split_metadata = MetadataHandler.load_split_metadata(
            split_path, metadata_dirs, subsets, modalities=['slide'])
        # Drop samples for which no data are available
        slides_ids = [
            os.path.splitext(feat_file)[0] for feat_dir in self.features_dirs for feat_file in os.listdir(feat_dir)
        ]
        split_metadata = split_metadata[split_metadata['slide_id'].isin(slides_ids)].reset_index(drop=True)
        # Drop duplicates
        if drop_duplicates == 'sample':
            split_metadata = split_metadata.drop_duplicates('slide_id', keep='first').reset_index(drop=True)
        elif drop_duplicates == 'case':
            split_metadata = split_metadata.drop_duplicates('case_id', keep='first').reset_index(drop=True)
        else:
            raise ValueError(f"Unknown level for dropping duplicates: {drop_duplicates}")
        self.split_metadata = split_metadata
        # Load patch metadata and data
        annotation_label_fn = partial(
            _binary_annotation_label_fn, target_class=self.label_col, other_thr=thresholds[0], target_thr=thresholds[1],
            include_unlabeled_samples=include_unlabeled_samples
        )
        self.feature_indices, self.patch_ids, self.patch_labels = \
            SlideDataHandler.load_patch_metadata(self.split_metadata, patches_dirs, patch_filters, annotation_label_fn)
        self.num_patches = len(torch.concat(self.patch_ids))
        print("Loading features into RAM")
        self.features = []
        for idx, row in tqdm(self.split_metadata.iterrows(), total=len(self.split_metadata)):
            source_id, slide_id = row[['source_id', 'slide_id']]
            features_path = os.path.join(self.features_dirs[source_id], slide_id)
            self.features.append(
                SlideDataHandler.load_features(features_path, self.feature_indices[idx], self.patch_ids[idx])
            )
        # Prepare data structures for sampling
        if self.sampling is None or self.sampling == 'uniform':
            self.slide_idx_list = torch.concat([
                torch.full((len(patch_ids),), slide_idx) for slide_idx, patch_ids in enumerate(self.patch_ids)
            ])
            self.feature_indices, self.patch_ids, self.patch_labels = \
                torch.concat(self.feature_indices), torch.concat(self.patch_ids), torch.concat(self.patch_labels)
            self.features = torch.concat(self.features)
            self.sampling_tree = None
        elif self.sampling == 'hierarchical_slide':
            self.slide_idx_list = None
            self.sampling_tree = {
                slide_idx: {
                    label: [
                        patch_idx for patch_idx, patch_label in enumerate(patch_labels) if patch_label == label
                    ]
                    for label in patch_labels.unique().tolist()
                }
                for slide_idx, patch_labels in enumerate(self.patch_labels)
            }
        elif self.sampling == 'hierarchical_label':
            self.slide_idx_list = None
            all_labels = torch.concat(self.patch_labels).unique().tolist()
            self.sampling_tree = {
                label: {
                    slide_idx: [
                        patch_idx for patch_idx, patch_label in enumerate(patch_labels) if patch_label == label
                    ]
                    for slide_idx, patch_labels in enumerate(self.patch_labels) if label in patch_labels
                }
                for label in all_labels
            }
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}")

    def get_metadata(self):
        return self.split_metadata

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        """
        :return: (dict)
            - 'features': (torch.Tensor) Feature vector of the patch.
            - 'target': (int) Prediction target of the patch.
            - 'patch_id': (int) Identifier of the patch.
            - 'slide_id': (list) Identifier of the slide.
            - 'source_id': (list) Identifier of the source.
        """
        if self.sampling is None:
            slide_idx = self.slide_idx_list[idx].item()
            patch_id = self.patch_ids[idx]
            features = self.features[idx]
            target = self.patch_labels[idx]
        elif self.sampling == 'uniform':
            patch_idx = torch.randint(len(self.patch_ids), (1,))[0].item()
            slide_idx = self.slide_idx_list[patch_idx].item()
            patch_id = self.patch_ids[idx]
            features = self.features[idx]
            target = self.patch_labels[patch_idx]
        elif self.sampling == 'hierarchical_slide':
            # Sample slide, target, and patch
            slide_idx = list(self.sampling_tree.keys())[
                torch.randint(len(self.sampling_tree), (1,))[0].item()
            ]
            target = list(self.sampling_tree[slide_idx].keys())[
                torch.randint(len(self.sampling_tree[slide_idx]), (1,))[0].item()
            ]
            patch_idx = self.sampling_tree[slide_idx][target][
                torch.randint(len(self.sampling_tree[slide_idx][target]), (1,))[0].item()
            ]
            # Sample patch
            patch_id = self.patch_ids[slide_idx][patch_idx]
            features = self.features[slide_idx][patch_idx]
            target = self.patch_labels[slide_idx][patch_idx]
        elif self.sampling == 'hierarchical_label':
            # Sample target, slide, and patch
            target = list(self.sampling_tree.keys())[
                torch.randint(len(self.sampling_tree), (1,))[0].item()
            ]
            slide_idx = list(self.sampling_tree[target].keys())[
                torch.randint(len(self.sampling_tree[target]), (1,))[0].item()
            ]
            patch_idx = self.sampling_tree[target][slide_idx][
                torch.randint(len(self.sampling_tree[target][slide_idx]), (1,))[0].item()
            ]
            # Sample patch
            patch_id = self.patch_ids[slide_idx][patch_idx]
            features = self.features[slide_idx][patch_idx]
            target = self.patch_labels[slide_idx][patch_idx]
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling}")
        source_id, slide_id = self.split_metadata.loc[slide_idx, ['source_id', 'slide_id']]
        sample_ids = {'patch_id': patch_id, 'slide_id': slide_id, 'source_id': source_id}
        return {'features': features, 'targets': target.unsqueeze(0), 'sample_ids': sample_ids}
