import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

from mnist.mnist_dataset import MNISTDataset
from typing import List
from pathlib import Path


class BinaryHopfieldNetwork(nn.Module):
    def __init__(self, input_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        # The 'synaptic weights'. Weights store the memory such that they correspond to a min-energy local minima state
        self.weights = torch.zeros((input_dim, input_dim))

    def learn(self, x: torch.Tensor):
        '''
        Essentially the 'forward pass'. Update the weights matrix s.t. the input data has minimal energy
        Uses hebbian learning to update the weights
        :param x: memory to store. (N x 1)
        '''
        self.weights = x @ x.T  # outer product
        self.weights.fill_diagonal_(0)

    def recall(self, x_noisy: torch.Tensor, steps: int = 10, save_intermediates: bool = False):
        '''
        Retreive a memory given a corrupted or partial input
        :param x_noisy: corrupted or partial input: (N x 1)
        :param steps: number of update steps to perform
        :param save_intermediates
        :return: associated memory (and maybe intermediate x's)
        '''
        energies = []
        ims = []
        for _ in range(steps):
            # pick a random neuron to update
            idx = torch.randint(0, self.input_dim, (1,))

            # if the sum of the weighted input neurons is not the same sign as the neuron to update,
            # then we flip its sign. Weights are determined from the memory
            activation = torch.sign(self.weights[idx, :] @ x_noisy)
            if torch.sign(x_noisy[idx]) != activation:
                x_noisy[idx] = activation

            if save_intermediates:

                ims.append(x_noisy.clone().reshape(28, 28))

            energy = -0.5 * (x_noisy.T @ self.weights) @ x_noisy
            energies.append(energy.item())

        return x_noisy, energies, ims


def denormalize_image(img: torch.Tensor):
    img[img <= 0] = 0
    img[img > 0] = 255.
    return img


def images2gif(ims: List[torch.Tensor], save_path: str = './output'):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)

    w, h = ims[0].shape[0], ims[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('./output/recall.mp4', fourcc, 1000, (w, h), False)

    # for i in range(len(ims)):
    #     ims[i] = to_pil_image(denormalize_image(ims[i])).numpy()

    for im in ims:
        frame = denormalize_image(im).to(torch.uint8).numpy()
        writer.write(frame)

    writer.release()


def run():
    dataset = MNISTDataset(download=True)
    img = dataset[torch.randint(0, len(dataset), (1,))][0]
    dim = len(img.flatten())

    hopfield_network = BinaryHopfieldNetwork(input_dim=dim)
    hopfield_network.learn(img.view(-1, 1))

    noisy_img = img.clone()
    corruption_size = 24
    noisy_img[:corruption_size, :corruption_size] = torch.randint(-1, 2, (corruption_size, corruption_size))
    noisy_img[noisy_img == 0] = 1

    denoised_img, energies, ims = hopfield_network.recall(noisy_img.view(-1, 1).clone(), steps=10000, save_intermediates=True)
    denoised_img = denoised_img.view(28, 28)

    # save a gif of the recall process
    images2gif(ims)

    fig, axs = plt.subplots(3, 1)
    axs[0].imshow(denormalize_image(noisy_img), cmap='gray', vmin=0, vmax=255)
    axs[1].imshow(denormalize_image(denoised_img), cmap='gray', vmin=0, vmax=255)
    axs[2].plot(list(range(len(energies))), energies)
    plt.show()


if __name__ == '__main__':
    run()
