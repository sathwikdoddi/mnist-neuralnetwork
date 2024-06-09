from mnist_database import get_mnist
import numpy as np
import matplotlib.pyplot as plt

images, labels = get_mnist()
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learn_rate = 0.02
num_correct = 0
iterations = 3
for iteration in range(iterations):
    for image, label in zip(images, labels):
        image.shape += (1,)
        label.shape += (1,)

        h_raw = b_i_h + w_i_h @ image
        h_activated = 1 / (1 + np.exp(-h_raw))

        o_raw = b_h_o + w_h_o @ h_activated
        o_activated = 1 / (1 + np.exp(-o_raw))

        e = 1 / len(o_activated) * np.sum((o_activated - label) ** 2, axis=0)
        num_correct += int(np.argmax(o_activated) == np.argmax(label))

        output_error = o_activated - label
        w_h_o -= learn_rate * output_error @ np.transpose(h_activated)
        b_h_o -= learn_rate * output_error

        hidden_error = np.transpose(w_h_o) @ output_error * (h_activated * (1 - h_activated))
        w_i_h -= learn_rate * hidden_error @ np.transpose(image)
        b_i_h -= learn_rate * hidden_error
    
    print(f"Accuracy: {round((num_correct / images.shape[0]) * 100, 2)}%")
    num_correct = 0

while True:
    sample = int(input("Enter a number from 0 to 59999 (-1 to quit): "))
    if sample == -1:
        break
        
    image = images[sample]
    plt.imshow(image.reshape(28,28), cmap="Greys")

    image.shape += (1,)

    h_raw = b_i_h + w_i_h @ image
    h_activated = 1 / (1 + np.exp(-h_raw))

    o_raw = b_h_o + w_h_o @ h_activated
    o_activated = 1 / (1 + np.exp(-o_raw))

    plt.title(f"Number Detected: {o_activated.argmax()}")
    plt.show()