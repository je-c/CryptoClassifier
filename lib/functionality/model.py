import torch
import torch.nn as nn
import torch.nn.functional as F

targetLabels = []
predsMade = []

def accuracy(outputs, labels):
    """
    Calculate model accuracy
        * :param outputs(torch.tensor): Pytorch weights tensor
        * :param labels(list(str)): List of known labels

    :return (torch.tensor): Prediction accuracy
    """
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class BaseClassifier(nn.Module):
    """
    BaseClassifier Class
    --------------------
    Extension of nn.module. Adds epoch step through methods for loss calculation by batch
    and prediction output storage
    """
    def calculate_loss(self, batch):
        """
        Perform a training step in the model.
            * :param batch(torch.DataLoader): Pytorch dataloader containing dataset

        :return loss(torch.tensor): Loss tensor
        """
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validate(self, batch):
        """
        Perform a validation step in the model.
            * :param batch(torch.DataLoader): Pytorch dataloader containing dataset

        :return (dict): Representation of loss and accuracy for the validation step
        """
        images, labels = batch 
        out = self(images)

        for i, j in zip(labels, out):
          targetLabels.append(i)
          predsMade.append(torch.argmax(j))
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {
            'val_loss': loss.detach(),
            'val_acc': acc
        }
        
    def validate_epoch(self, outputs):
        """
        Accumulates validation steps completed in a single epoch. 
            * :param outputs(dict): Dictionary of loss and accuracy statistics for each validation step

        :return (dict): Validation results
        """
        return {
            'val_loss': torch.stack([i['val_loss'] for i in outputs]).mean().item(), 
            'val_acc': torch.stack([i['val_acc'] for i in outputs]).mean().item()
        }
    
    def epoch_wrapup(self, epoch, results):
        """
        Outputs epoch statistics 
            * :param epoch(int): Epoch #
            * :param results(dict): Dictionary of statistics for the epoch

        :return (NoneType): None
        """
        print(f"Epoch [{epoch}]") 
        print(f"    - last_lr: {results['lrs'][-1]:.8f}")
        print(f"    - train_loss: {results['train_loss']:.4f}")
        print(f"    - val_loss: {results['val_loss']:.4f}")
        print(f"    - val_acc: {results['val_acc']:.4f}")

def conv_block(in_channels, out_channels, pool=False):
    """
    Convolution layer structure. Convolves incoming image, performs batch normalisation and applys ReLU.
    Optionally supports layer pooling
        * :param in_channels(int): Expected number of incoming channels
        * :param out_channels(int): Output number channel
        * :param pool(bool): Invokes layer pooling

    :return (torch.tensor): Linearised convolution layer
    """
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True)
    ]

    if pool: layers.append(nn.MaxPool2d(3))
    
    return nn.Sequential(*layers)

class ResNet9(BaseClassifier):
    """
    ResNet9 Class
    -------------

    Extends BaseClassifier with Residual Neural Net structures. Handles weights forward 
    pass and structures the network.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128)
        self.res1 = nn.Sequential(
                        conv_block(128, 128), 
                        conv_block(128, 128)
                    )
        
        self.conv3 = conv_block(128, 256)
        self.conv4 = conv_block(256, 512, pool = True)
        self.res2 = nn.Sequential(
                        conv_block(512, 512), 
                        conv_block(512, 512)
                    )

        self.classifier = nn.Sequential(
                                nn.MaxPool2d(3),
                                nn.Flatten(),
                                nn.Linear(512, num_classes)
                          )
        
    def forward(self, X):
        """
        Convolution forward pass. Convolves image through network, computes residual layers and final
        classifier
            * :param X(torch.tensor): Incoming data

        :return out(torch.tensor): Convolved network
        """
        out = self.conv1(X)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

@torch.no_grad()
def evaluate(model, validation_dl):
    """
    Evaluate model at epoch
        * :param model(torch.nn.module): ResNet model
        * :param validation_dl(torch.dataloader): Validation data

    :return (dict): Validation results for model at current epoch
    """
    model.eval()
    return model.validate_epoch([model.validate(batch) for batch in validation_dl])

def get_lr(optimizer):
    """
    Return current learning rate from optimiser function
        * :param optimiser(torch.optim): Optimiser function

    :return (float): Learning rate at current epoch
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit(params, model, training_dl, validation_dl, optimiser_f=torch.optim.SGD):
    """
    Main training function. Compiles and trains the model on given data. Handles
    learning rate sheduling and validation statistics storage.
        * :param params(dict): Model parameters
        * :param model(torch.nn.module): Model
        * :param training_dl(torch.dataloader): Training data
        * :param validation_dl(torch.dataloader): Validation data
        * :param optimiser_f(torch.optim): Optimiser function

    :return history(list): Statistics for each epoch training/validation step
    """
    epochs, max_lr, weight_decay, grad_clip = [params[key] for key in params]
    torch.cuda.empty_cache()
    history = []
    
    optimiser = optimiser_f(
                    model.parameters(), 
                    max_lr, 
                    weight_decay=weight_decay
                )
    sched = torch.optim.lr_scheduler.OneCycleLR(
                    optimiser, max_lr, 
                    epochs=epochs,
                    steps_per_epoch=len(training_dl)
            )
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in training_dl:
            loss = model.calculate_loss(batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimiser.step()
            optimiser.zero_grad()
            
            lrs.append(get_lr(optimiser))
            sched.step()
        
        # Validate epoch
        result = evaluate(model, validation_dl)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_wrapup(epoch, result)
        history.append(result)

    return history

