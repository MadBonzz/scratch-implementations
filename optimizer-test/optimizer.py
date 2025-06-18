import torch
from torch.optim import Optimizer

class CustomAdam(Optimizer):
    def __init__(self, params, stepsize = 0.001, bias_m1 = 0.9, bias_m2 = 0.999, epsilon = 1e-8, bias_correction = True):
        DEFAULTS = dict(stepsize = stepsize, bias_m1 = bias_m1, bias_m2 = bias_m2, epsilon = epsilon, bias_correction = bias_correction)
        #Initialize the optimizer
        super(CustomAdam, self).__init__(params, DEFAULTS)

    def step(self, closure = None):
        #Set loss to none
        loss = None
        #Check if this is the first step - if not, increment the current step
        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1
        #Iterate over "groups" of parameters (layers of parameters in the network) to begin processing and computing the next set of params
        for param_group in self.param_groups:
            #Iterate over individual parameters
            for param in param_group["params"]:
                #Check if gradients have been computed for each parameter
                #If not - if there are no gradients - then skip the parameter
                if param.grad.data == None:
                    continue
                else: gradients = param.grad.data
                #Use Adam optimization method - first, define all the required arguments for the parameter if we are on the first step
                state = self.state[param]
                if len(state) == 0:
                    #Set the first and second moment estimates to zeroes
                    self.state["step"] = 0
                    self.state["first_moment_estimate"] = torch.zeros_like(param.data)
                    self.state["second_moment_estimate"] = torch.zeros_like(param.data)
                self.state["step"] += 1
                #Declare variables from state - inplace methods modify state variable directly
                first_moment_estimate = self.state["first_moment_estimate"]
                second_moment_estimate = self.state["second_moment_estimate"]
                #Compute the first moment estimate - B_1 * m_t + (1-B_1) * grad (uncentered)
                first_moment_estimate = first_moment_estimate.mul(param_group["bias_m1"]).add(gradients * (1.0 - param_group["bias_m1"]))
                #Compute the second moment estimate - B_2 * v_t + (1-B_2) * grad^2 (uncentered)
                second_moment_estimate = second_moment_estimate.mul(param_group["bias_m2"]).add(gradients.pow_(2) * (1.0 - param_group["bias_m2"]))
                #Perform bias correction if parameter is set to true
                if param_group["bias_correction"]:
                    #Perform bias correction for the first moment estimate: m_t / (1 -B_1^t)
                    first_moment_estimate.divide_(1.0 - (param_group["bias_m1"] ** self.state["step"]))
                    #Perform bias correction for second moment estimate: v_t / (1 - B_2^t)
                    second_moment_estimate.divide_(1.0 - (param_group["bias_m2"] ** self.state["step"]))
                
                self.state["first_moment_estimate"].mul_((self.state["step"] - 1) / self.state["step"]).add_((1 / self.state["step"]) * first_moment_estimate)
                self.state["second_moment_estimate"].mul_((self.state["step"] - 1) / self.state["step"]).add_((1 / self.state["step"]) * second_moment_estimate)
                param.data.add_((-param_group["stepsize"]) * first_moment_estimate.divide_(second_moment_estimate.sqrt_() + param_group["epsilon"]))
        return loss