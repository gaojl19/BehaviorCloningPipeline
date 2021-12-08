import torch
import numpy as np


if __name__ == "__main__":

    target = np.array([
            [-0.9536,  0.9672,  0.2525, -0.9651]])
    actions = np.array([
            [0.0000, 0.0000, 0.1048, 0.0000]])

    input = torch.autograd.Variable(torch.from_numpy(actions)).float()
    output = torch.autograd.Variable(torch.from_numpy(target)).float()

    print("predicts: ", input)
    print("actual: ", output)
    mse = torch.nn.MSELoss()
    loss = mse(input, output)
    print("loss: ", loss)

