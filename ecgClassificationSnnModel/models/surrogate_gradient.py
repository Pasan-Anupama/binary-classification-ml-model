import torch

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        slope = 10.0
        grad_input = grad_output.clone()
        surrogate_grad = slope * torch.clamp(1 - input.abs(), min=0)
        return grad_input * surrogate_grad
