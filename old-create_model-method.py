def _create_model(self, arch, hidden_units):
        """
        Creates and returns a deep learning model based on the specified architecture 
        and hidden units, with pretrained weights and a new classifier for VGG16 or ResNet50.

        Args:
            arch (str): Model architecture to use (e.g., 'vgg16' or 'resnet50').
            hidden_units (int): Number of hidden units for the model's classifier.

        Returns:
            torch.nn.Module: The model with the updated classifier.
        """
        # Normalise the architecture name to handle variations like 'VGG16'
        # and 'vgg16'.
        # TODO: May be redundant but currently to scared to change anything...
        arch = arch.lower()


        if hasattr(models, arch):
        # TODO: Potential redundant code that could be refactored.
            if arch == 'vgg16':
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

                # Freeze parameters to stop backpropagation
                for param in model.parameters():
                    param.requires_grad = False

                # Create a new classifier for VGG16
                classifier = nn.Sequential(
                    nn.Linear(model.classifier[0].in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, len(self.class_to_idx)),
                    nn.LogSoftmax(dim=1)
                )
                model.classifier = classifier

            elif arch == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                
                # Freeze parameters to stop backpropagation
                for param in model.parameters():
                    param.requires_grad = False

                # New classifier for ResNet.
                # ResNet50 uses `fc` instead of `classifier`
                model.fc = nn.Sequential(
                    nn.Linear(model.fc.in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, len(self.class_to_idx)),
                    nn.LogSoftmax(dim=1)
                )
            else:
                raise ValueError(f"Architecture {arch} not supported or weights not available.")
        else:
            # For older versions of torchvision that use 'pretrained=True'
            # instead of weights
            # TODO: May be redundant/better way to handle this.
            model = getattr(models, arch)(pretrained=True)

        print(f"Classifier model loaded with architecture {arch} and hidden units {hidden_units}.")

        return model