from torch.utils.data import DataLoader

from .patch import PatchFeatureDataset


class DatasetFactory:

    @staticmethod
    def build(dataset_args, model_args):

        if model_args['mode'] == 'patch-segmentation':

            if dataset_args.get('train_subsets') is not None:
                train_dataset = DatasetFactory._build_patch_dataset(dataset_args, 'train')
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=dataset_args['train_batch_size'],
                    shuffle=True
                )
            else:
                train_dataset, train_loader = None, None

            if dataset_args.get('val_subsets') is not None:
                val_dataset = DatasetFactory._build_patch_dataset(dataset_args, 'val')
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=dataset_args['val_batch_size'],
                    shuffle=False
                )
            else:
                val_dataset, val_loader = None, None

            if dataset_args.get('test_subsets') is not None:
                test_dataset = DatasetFactory._build_patch_dataset(dataset_args, 'test')
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=dataset_args['val_batch_size'],
                    shuffle=False
                )
            else:
                test_dataset, test_loader = None, None

        else:
            raise ValueError(f"Unknown mode: {dataset_args['mode']}")

        return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader

    @staticmethod
    def _build_patch_dataset(args, stage):

        if stage == 'train':
            subsets = args['train_subsets']
            sampling = args['sampler']
        elif stage == 'val':
            subsets = args['val_subsets']
            sampling = None
        elif stage == 'test':
            subsets = args['test_subsets']
            sampling = None
        else:
            raise ValueError(f"Unknown stage: {stage}")

        dataset = PatchFeatureDataset(
            split_path=args['split_path'],
            metadata_dirs=args['metadata_dirs'],
            subsets=subsets,
            patches_dirs=args['patches_dirs'],
            features_dirs=args['features_dirs'],
            label_cols=args.get('targets', None),
            thresholds=tuple(args.get('annotation_thresholds', [0.0, 0.0])),
            patch_filters=args.get('patch_filters', None),
            drop_duplicates=args.get('drop_duplicates', 'sample'),
            sampling=sampling,
            include_unlabeled_samples=args.get('include_unlabeled_samples', False),
        )

        return dataset
