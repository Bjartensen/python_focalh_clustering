import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



class Train():
    """
    Class for training a nn model like UNet
    """
    def __init__(self, model, image_crit, learning_rate, momentum):
        self.model = model
        self.image_criterion = image_crit
        self.lr = learning_rate
        self.momentum = momentum
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # Model? Data?

    def run(self, epochs, events, targets):
        """
        Main program that runs for some number of epochs.
        Could be extended to take in eval data just to compute eval loss.
        """
        Nevents = len(events)
        print(f"Training on {Nevents} events.")
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            self.optimizer.zero_grad()
            image_outputs = self.model(events)
            image_loss = self.image_criterion(image_outputs, targets)
            total_loss = image_loss# + count_loss
            total_loss.backward()
            self.optimizer.step()
            print(total_loss.item())
