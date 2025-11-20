import os
import json
import argparse
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets import DatasetFactory
from models import ModelFactory
from training import Callback, train_classification_model, test_classification_model


def get_args():
    parser = argparse.ArgumentParser()

    # Modalities
    parser.add_argument('--mode', type=str, default='patch-segmentation',
                        choices=['patch-segmentation'])

    # Loading and saving
    parser.add_argument('--split-path', type=str, required=True)
    parser.add_argument('--metadata-dirs', type=str, nargs='+', required=True)
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--patches-dirs', type=str, nargs='+')
    parser.add_argument('--features-dirs', type=str, nargs='+')

    # Dataset args
    parser.add_argument('--train-subsets', default=['train'], nargs='+', type=str,
                        help='Split subsets that are used for training.')
    parser.add_argument('--val-subsets', default=['test'], nargs='+', type=str,
                        help='Split subsets that are used for validation.')
    parser.add_argument('--test-subsets', default=None, nargs='+', type=str,
                        help='Split subsets that are used for testing.')
    parser.add_argument('--drop-duplicates', type=str, default='sample', choices=['sample', 'case'])
    parser.add_argument('--sampler', type=str, default=None,
                        help='Sampling strategy for the training data. If None, random shuffling will be used.')
    parser.add_argument('--patch-filters', default=None,
                        help="Filters to only use a selected subset of patches per slide."
                             "Pass {'has_annot': [1, 2]} to only use patches with some annotation of class 1 or 2."
                             "Pass {'exclude_annot': [0, 8]} to only use patches with no annotation of class 0 and 8.")
    parser.add_argument('--annotation-thresholds', type=float, nargs='+', default=[0.0, 0.0],
                        help="Thresholds to derive binary patch labels from the annotation fractions of the patch."
                             "If annot_frac <= thr[0], the label is 0. If annot_frac > thr[1], the label is 1."
                             "Otherwise, the patch is dropped.")

    # Model args
    parser.add_argument('--input-dim', type=int, default=2048,
                        help="The dimension of the feature vectors.")
    parser.add_argument('--num-classes', type=int, default=2,
                        help="The number of classes to predict.")
    parser.add_argument('--targets', nargs='+', type=str, default=['label'],
                        help="The target labels to predict.")
    parser.add_argument('--hidden-depth', type=int, default=0,
                        help="The number of hidden layers of fully connected models, e.g., patch classifiers and MLPs.")
    parser.add_argument('--dropout', type=float, default=None,
                        help="Fraction of neurons to drop per targeted layer. None to apply no dropout.")
    parser.add_argument('--model-type', type=str, default='mlp', choices=['mlp', 'snn'],
                        help="The type of model to train.")
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help="The dimension of the latent vectors in the network.")

    # Training args
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--schedule-lr', action='store_true')
    parser.add_argument('--objective', type=str, default='cross-entropy')
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=1)
    parser.add_argument('--early-stopping', action='store_true')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--grad-clip', type=float, default=None)

    # Testing args
    parser.add_argument('--test-checkpoint', type=str, default='best', choices=['best', 'last'])

    # Environment args
    parser.add_argument('--device', type=str, default='cpu')

    # Parse all args
    args = parser.parse_args()
    if args.patch_filters is not None:
        args.patch_filters = json.loads(args.patch_filters)

    return args


def main(args=None):
    # Process args and create file structures
    args = get_args() if args is None else args
    print(json.dumps(vars(args), indent=4))
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(save_dir)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Set up environment
    device = torch.device(args.device)
    tb_writer = SummaryWriter(save_dir)

    # Set up dataset structures
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = \
        DatasetFactory.build(vars(args), vars(args))

    # Set up model and classifier
    model, classifier = ModelFactory.build(vars(args), device)

    # Set up callback
    callback = Callback(
        schedule_lr=args.schedule_lr, checkpoint_epoch=args.val_interval, path_checkpoints=save_dir,
        early_stop=args.early_stopping)

    # Run training
    print("Model training")
    train_classification_model(
        model=model, classifier=classifier, optimizer=classifier.optimizer, n_epochs=args.num_epochs,
        lr_init=args.learning_rate, dataloader_train=train_loader, dataloader_val=val_loader, callback=callback,
        label_cols=args.targets, n_epoch_val=args.val_interval, tb_writer=tb_writer, verbose=False)

    # Apply trained model to test set
    if test_loader is not None:
        print(f"Test set evaluation with checkpoint: {args.test_checkpoint}")
        model = callback.load_checkpoint(model, checkpoint=args.test_checkpoint)
        test_classification_model(
            model=model, classifier=classifier, dataloader_test=test_loader, callback=callback, label_cols=args.targets,
            tb_writer=tb_writer, verbose=False)

    # Clean up
    tb_writer.close()
    print("Finished model training")


if __name__ == '__main__':
    main()
