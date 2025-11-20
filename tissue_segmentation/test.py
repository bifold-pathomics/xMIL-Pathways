import os
import json
import argparse

import torch

from datasets import DatasetFactory
from models import ModelFactory
from training import Callback, test_classification_model


def get_args():
    parser = argparse.ArgumentParser()

    # Loading and saving
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--test-checkpoint', type=str, default=None, choices=[None, 'best', 'last'])
    parser.add_argument('--split-path', type=str, required=True)
    parser.add_argument('--metadata-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--results-dir', type=str, required=True)

    # - WSI data
    parser.add_argument('--patches-dirs', type=str, nargs='+')
    parser.add_argument('--features-dirs', type=str, nargs='+')

    # Dataset args
    parser.add_argument('--test-subsets', default=None, nargs='+', type=str,
                        help='Split subsets that are used for testing.')
    parser.add_argument('--drop-duplicates', type=str, default='sample', choices=['sample', 'case'])
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--include-unlabeled-samples', action='store_true')

    # - WSI dataset
    parser.add_argument('--patch-filters', default=None,
                        help="Filters to only use a selected subset of patches per slide."
                             "Pass {'has_annot': [1, 2]} to only use patches with some annotation of class 1 or 2."
                             "Pass {'exclude_annot': [0, 8]} to only use patches with no annotation of class 0 and 8.")
    parser.add_argument('--max-bag-size', type=int, default=None,
                        help="Maximum number of patches per slide. Slides with more patches are dropped.")
    parser.add_argument('--preload-data', action='store_true',
                        help="Whether to preload all features into RAM before starting training.")

    # - Patch segmentation dataset
    parser.add_argument('--annotation-thresholds', type=float, nargs='+', default=[0.0, 0.0],
                        help="Thresholds to derive binary patch labels from the annotation fractions of the patch."
                             "If annot_frac <= thr[0], the label is 0. If annot_frac > thr[1], the label is 1."
                             "Otherwise, the patch is dropped.")

    # Environment args
    parser.add_argument('--device', type=str, default='cpu')

    # Parse all args
    args = parser.parse_args()

    if args.patch_filters is not None:
        args.patch_filters = json.loads(args.patch_filters)

    return args


def main(args=None):
    # Process and save input args
    args = get_args() if args is None else args
    print(json.dumps(vars(args), indent=4))
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=False)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set up environment
    device = torch.device(args.device)

    # Load args from model training
    with open(os.path.join(args.model_dir, 'args.json')) as f:
        model_args = json.load(f)

    # Derive dataset args from given args (dominant) and model args (default)
    dataset_args = {**model_args, **vars(args), **{'train_subsets': None, 'val_subsets': None}}

    # Load dataset structures
    _, _, _, _, test_dataset, test_loader = DatasetFactory.build(dataset_args, model_args)

    # Set up callback, model, and load model weights
    callback = Callback(
        schedule_lr=None, checkpoint_epoch=1, path_checkpoints=args.model_dir, early_stop=False, device=device,
        results_dir=save_dir)

    model, classifier = ModelFactory.build(model_args, device)

    print(f"Loading model into RAM from: {args.model_dir}")
    checkpoint = args.test_checkpoint if args.test_checkpoint is not None else model_args['test_checkpoint']
    model = callback.load_checkpoint(model, checkpoint=checkpoint)

    # Run test

    print(f"Test set evaluation with checkpoint: {checkpoint}")
    test_classification_model(
        model=model, classifier=classifier, dataloader_test=test_loader, callback=callback,
        label_cols=model_args.get('targets', ['label']), tb_writer=None, verbose=False)


if __name__ == '__main__':
    main()
