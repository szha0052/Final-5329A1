import numpy as np

class MiniBatchFit:
    def __init__(self, model, optimizer, 
                 X_train, Y_train, 
                 output_dim, 
                 num_epochs=10, 
                 batch_size=32, 
                 learning_rate=0.01,
                 write = True):
        """
        :param model: Your model object, which should include methods like forward, compute_loss, backward, update, etc.
        :param optimizer: Optimizer object, in this example, only optimizer.increment_time_step() is used
        :param X_train: Training features, shape (N, feature_dim)
        :param Y_train: Training labels, shape (N, )
        :param output_dim: Output dimension, used to generate one-hot labels
        :param num_epochs: Number of training epochs
        :param batch_size: Batch size
        :param learning_rate: Learning rate
        :param velocity: Momentum (if the model update uses momentum-related mechanisms)
        """
        self.model = model
        self.optimizer = optimizer
        self.X_train = X_train
        self.Y_train = Y_train
        self.output_dim = output_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.write = write

    def fit(self):
        for epoch in range(self.num_epochs):
            # If your optimizer needs to increment the time step, call it here
            if hasattr(self.optimizer, 'increment_time_step'):
                # Assume the optimizer has an increment_time_step method
                # to update the time step or other states
                self.optimizer.increment_time_step()

            epoch_loss = 0
            # Iterate through the data in batches
            for i in range(0, len(self.X_train), self.batch_size):
                X_batch = self.X_train[i:i+self.batch_size]
                Y_batch = self.Y_train[i:i+self.batch_size]

                # Generate one-hot encoding
                Y_batch_one_hot = np.eye(self.output_dim)[Y_batch]

                # Forward propagation
                logits = self.model.forward(X_batch, training=True)

                # Compute loss
                loss = self.model.compute_loss(logits, Y_batch_one_hot)
                epoch_loss += loss

                # Backward propagation
                self.model.backward()

                # Update parameters
                self.model.update(self.learning_rate)

            # Compute and print the average loss for each epoch
            avg_loss = epoch_loss / (len(self.X_train) // self.batch_size)
            if self.write:
                # Assume your model has a write method to log the training process
                print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}')

