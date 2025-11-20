from .mlp import ClassificationMLP, BinaryMLPClassifier


class ModelFactory:

    @staticmethod
    def build(model_args, device):

        # Process args
        num_targets = len(model_args.get('targets', ['label']))

        if model_args['mode'] == 'patch-segmentation':

            model = ClassificationMLP(
                input_dim=model_args['input_dim'],
                hidden_dim=model_args['hidden_dim'],
                hidden_depth=model_args['hidden_depth'],
                num_classes=model_args['num_classes'],
                num_targets=num_targets,
                dropout=model_args['dropout'],
                snn=(model_args['model_type'] == 'snn')
            )
            classifier = BinaryMLPClassifier(
                model=model,
                learning_rate=model_args['learning_rate'],
                weight_decay=model_args['weight_decay'],
                objective=model_args['objective'],
                gradient_clip=model_args['grad_clip'],
                device=device
            )

        else:
            raise ValueError(f"Unknown mode: {model_args['mode']}")

        return model, classifier
