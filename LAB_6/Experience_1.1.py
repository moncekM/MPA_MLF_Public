import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.metric import Accuracy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mxnet.gluon.data.vision import CIFAR10

font = {'weight': 'bold', 'size': 12}
matplotlib.rc('font', **font)

# Load CIFAR-10 dataset
transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

cifar_train = CIFAR10(train=True).transform_first(transform_fn)
cifar_test = CIFAR10(train=False).transform_first(transform_fn)

train_data = gluon.data.DataLoader(cifar_train, batch_size=25, shuffle=True)
test_data = gluon.data.DataLoader(cifar_test, batch_size=25, shuffle=False)

labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
          5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def display_channels_separately(image: np.array) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
    axs[0].imshow(image[:, :, 0], cmap='Reds')
    axs[1].imshow(image[:, :, 1], cmap='Blues')
    axs[2].imshow(image[:, :, 2], cmap='Greens')
    axs[3].imshow(image)
    plt.show()

# Build the model in MXNet
net = nn.Sequential()
net.add(
    nn.Flatten(),
    nn.Dense(128, activation='sigmoid'),
    nn.Dense(10, activation='relu'),
    nn.Dense(128, activation='relu'),
    nn.Dense(10)
)

ctx = mx.cpu()  # or mx.gpu() if you have GPU
net.initialize(mx.init.Xavier(), ctx=ctx)

loss_fn = SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.001})

# Training loop
epochs = 50
for epoch in range(epochs):
    train_loss = 0.
    train_acc = Accuracy()
    for data, label in train_data:
        data = data.as_in_ctx(ctx)
        label = label.as_in_ctx(ctx)
        with autograd.record():
            output = net(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        train_loss += loss.mean().asscalar()
        train_acc.update(label, output)

    print(f'Epoch {epoch + 1}: loss {train_loss / len(train_data):.4f}, acc {train_acc.get()[1] * 100:.2f}%')

# Evaluate on test data
test_acc = Accuracy()
for data, label in test_data:
    data = data.as_in_ctx(ctx)
    label = label.as_in_ctx(ctx)
    outputs = net(data)
    test_acc.update(label, outputs)

print(f'Test Accuracy: {test_acc.get()[1] * 100:.2f} %')

# Visualizing predictions on test set

def show_the_best_predictions(model, test_loader, n_of_pred=10):
    all_probs = []
    all_labels = []
    all_images = []

    for data, label in test_loader:
        data = data.as_in_ctx(ctx)
        outputs = model(data)
        probs = nd.softmax(outputs)
        all_probs.append(probs.asnumpy())
        all_labels.append(label.asnumpy())
        all_images.append(data.transpose((0, 2, 3, 1)).asnumpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_images = np.concatenate(all_images, axis=0)

    pred_classes = np.argmax(all_probs, axis=1)
    correct_idx = np.where(pred_classes == all_labels)[0]
    correct_probs = all_probs[correct_idx, all_labels[correct_idx]]

    best_idx = correct_idx[np.argsort(correct_probs)[::-1][:n_of_pred]]

    images = all_images[best_idx]
    probs = correct_probs[np.argsort(correct_probs)[::-1][:n_of_pred]]
    labels_pred = pred_classes[best_idx]

    concat_img = np.concatenate(images, axis=1)
    plt.figure(figsize=(20, 10))
    plt.imshow(concat_img)
    for i in range(n_of_pred):
        text = f'{labels[labels_pred[i]]}: {probs[i] * 100:.2f} %'
        plt.text((32 / 2) + 32 * i - len(labels[labels_pred[i]]), 32 * (5 / 4), text)
    plt.axis('off')
    plt.show()

show_the_best_predictions(net, test_data, n_of_pred=10)
