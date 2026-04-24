from math import exp

import Forge
from utils import load_data_to_forge

# Download the training and test set from:
# https://github.com/kellen-sun/Exploring_AI/blob/main/MNIST/mnist_train.bin
# https://github.com/kellen-sun/Exploring_AI/blob/main/MNIST/mnist_test.bin
# The format is: 3 integers (4 bytes) for n, r, c (r=c=28)
# Then n*r*c Bytes (0-255) of grayscale values for the images
# n = 60k for the training set, 10k for test
# Then labels (the digit from 0-9), 1 Byte each

train_x, train_y, shape = load_data_to_forge("./py/examples/mnist_train.bin")
print(f"Loaded {shape[0]} training images of size {shape[1]}x{shape[2]}")
test_x, test_y, test_shape = load_data_to_forge("./py/examples/mnist_test.bin")
print(f"Loaded {test_shape[0]} test images of size {test_shape[1]}x{test_shape[2]}")

# We'll create a MLP model with two hidden layers of 16 nodes
# and use the tanh activation function into a final softmax
# then gather a final cross-entropy loss

# Our parameters:
Forge.set_seed(42)
bias_scale = 0.00001
# Xavier scale init weights
W1 = (Forge.rand(64, 784) - 0.5) * 0.1682
W2 = (Forge.rand(64, 64) - 0.5) * 0.4330
W3 = (Forge.rand(10, 64) - 0.5) * 0.5694
B1 = (Forge.rand(64) - 0.5) * bias_scale
B2 = (Forge.rand(64) - 0.5) * bias_scale
B3 = (Forge.rand(10) - 0.5) * bias_scale


def forward(A0):
    """Given some inputs A0, we calculate the activations after each layer"""
    global W1, W2, W3, B1, B2, B3
    A1 = Forge.tanh(A0 @ W1.T + B1)
    A2 = Forge.tanh(A1 @ W2.T + B2)
    A3 = A2 @ W3.T + B3
    # softmax
    exp_A3 = Forge.exp(A3)
    P = exp_A3 / exp_A3.sum(axis=1, keepdims=True)
    return P, A3, A2, A1


def accuracy(Ps, GTs):
    acc = [
        max(range(10), key=lambda k: i[k]) == max(range(10), key=lambda k: j[k])
        for i, j in zip(Ps.list(), GTs.list())
    ]
    return sum(acc) / len(acc)


def total_loss(Ps, GTs):
    """Given predictions and ground truths, what's our accuracy and loss?"""
    acc = accuracy(Ps, GTs)
    loss = (-GTs * (Ps + 1e-9).log()).sum()
    return loss.list() / len(Ps), acc


def backward(Ps, A3s, A2s, A1s, GTs, A0s):
    global W1, W2, W3
    n = len(GTs)  # batch size

    dB3 = Ps - GTs
    dW3 = (dB3.T @ A2s) / n
    # layer 2
    dA2 = dB3 @ W3
    dB2 = (1.0 - (A2s * A2s)) * dA2
    dW2 = (dB2.T @ A1s) / n
    # layer 1
    dA1 = dB2 @ W2
    dB1 = (1.0 - (A1s * A1s)) * dA1
    dW1 = (dB1.T @ A0s) / n
    # biases
    d_Bias1 = dB1.sum(axis=0) / n
    d_Bias2 = dB2.sum(axis=0) / n
    d_Bias3 = dB3.sum(axis=0) / n

    return dW1, dW2, dW3, d_Bias1, d_Bias2, d_Bias3


def train(batchsize=1, alpha=0.1, epochs=10, view=100000, expo=True):
    global train_x, train_y
    global W1, W2, W3, B1, B2, B3

    batchcount = len(train_x) // batchsize
    test()

    for i in range(epochs):
        a = alpha * exp(-i * 0.3) if expo else alpha * (1 - i / epochs)
        for j in range(batchcount):
            start = j * batchsize
            end = start + batchsize

            # Grab batch
            batch_x = train_x[start:end]
            batch_y = train_y[start:end]

            # Forward pass
            P, A3, A2, A1 = forward(batch_x)

            # Backwards pass
            dW1, dW2, dW3, dB1, dB2, dB3 = backward(P, A3, A2, A1, batch_y, batch_x)

            # Update weights
            decay = 0.001
            W1 -= (dW1 + decay * W1) * a
            W2 -= (dW2 + decay * W2) * a
            W3 -= (dW3 + decay * W3) * a
            B1 -= dB1 * a
            B2 -= dB2 * a
            B3 -= dB3 * a
            if j % view == view - 1:
                test()
        test()


def test():
    global test_x, test_y, train_x, train_y
    P, A3, A2, A1 = forward(test_x)
    loss_metrics = total_loss(P, test_y)
    P, A3, A2, A1 = forward(train_x)
    train_loss_metrics = total_loss(P, train_y)

    print(
        f"Test Loss: {loss_metrics[0]:.4f}, Test Accuracy: {100*loss_metrics[1]:.2f}%"
    )
    print(
        f"Train Loss: {train_loss_metrics[0]:.4f}, Train Accuracy: {100*train_loss_metrics[1]:.2f}%"
    )


train(batchsize=500, alpha=0.2, epochs=25, expo=False)
test()
