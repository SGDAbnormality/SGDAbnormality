import numpy as np
import torch
from tqdm import tqdm

from train_eval import get_loss

"""# Loss Function"""

"""# Training Set-up"""


def rate_function(model, s_values, device, data_loader):
  ret = []
  losses = get_loss(device, model, data_loader)
  with tqdm(total=len(s_values)) as pbar:
      for s_value in s_values:
          ret.append(rate_function_single_value(losses, s_value, device))
          pbar.update(1)
  return np.array(ret)

def inv_rate_function(model, s_values, device, data_loader):
  ret = []
  losses = get_loss(device, model, data_loader)
  with tqdm(total=len(s_values)) as pbar:
      for s_value in s_values:
          ret.append(inv_rate_function_single_value(model, losses, s_value))
          pbar.update(1)
  return np.array(ret)

def rate_function_single_value(losses, s_value, device):

  min_lamb=torch.tensor(0).to(device)
  max_lamb=torch.tensor(100000).to(device)

  s_value=torch.tensor(s_value, dtype=torch.float32).to(device)
  return aux_rate_function_TernarySearch(losses, s_value, min_lamb, max_lamb, 0.001, device)

def inv_rate_function_single_value(losses, s_value, device):

  min_lamb=torch.tensor(0).to(device)
  max_lamb=torch.tensor(100000).to(device)

  s_value=torch.tensor(s_value, dtype=torch.float32).to(device)

  return aux_inv_rate_function_TernarySearch(losses, s_value, min_lamb, max_lamb, 0.001, device)





def jensen_val(losses, lamb, sign, device):
    return (
        torch.logsumexp(-sign.detach()*lamb * losses, 0)
        - torch.log(torch.tensor(losses.shape[0], dtype=torch.float32, device=device).requires_grad_(True))
        + sign.detach()*lamb * torch.mean(losses)
    )

def eval_rate_at_lambda(losses, lamb, s_value, device):
    return (lamb * torch.abs(s_value) - jensen_val(losses, lamb, torch.sign(s_value).detach(), device))

def eval_rate_at_lambda_signed(losses, lamb, s_value, device):
    return torch.sign(s_value.detach()) * eval_rate_at_lambda(losses, lamb, s_value, device)

def eval_inverse_rate_at_lambda(losses, lamb, s_value, device):
    return  (torch.abs(s_value) + jensen_val(losses, lamb, torch.sign(s_value).detach(), device))/lamb

def eval_inverse_rate_at_lambda_signed(losses, lamb, s_value, device):
    return torch.sign(s_value.detach()) * eval_inverse_rate_at_lambda(losses, lamb, s_value, device)

def aux_rate_function_TernarySearch(losses, s_value, low, high, epsilon, device):

    iter = 0
    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3

        if eval_rate_at_lambda(losses, mid1, s_value, device) < eval_rate_at_lambda(losses, mid2, s_value, device):
            low = mid1
        else:
            high = mid2

        iter += 1
        if iter > 1000:
            print("EOEOEOEOEOEOEOEOEOEOEOEOEOEOE")
            break

    # Return the midpoint of the final range
    mid = (low + high) / 2
    return [
        (eval_rate_at_lambda_signed(losses, mid, s_value, device)).detach().cpu().numpy(),
        mid.detach().cpu().numpy(),
        jensen_val(losses,mid,torch.sign(s_value),device).detach().cpu().numpy(),
    ]

def aux_inv_rate_function_TernarySearch(losses, s_value, low, high, epsilon, device):
    iter = 0

    while (high - low) > epsilon:
        mid1 = low + (high - low) / 3
        mid2 = high - (high - low) / 3
        if eval_inverse_rate_at_lambda(losses, mid1, s_value, device) < eval_inverse_rate_at_lambda(losses, mid2, s_value, device):
            high = mid2
        else:
            low = mid1

        iter += 1
        if iter > 1000:
            print("EOEOEOEOEOEOEOEOEOEOEOEOEOEOE")
            break

    # Return the midpoint of the final range
    mid = (low + high) / 2
    return [
        eval_inverse_rate_at_lambda_signed(losses, mid, s_value, device).detach().cpu().numpy(),
        mid.detach().cpu().numpy(),
        jensen_val(losses, mid, torch.sign(s_value), device).detach().cpu().numpy()
    ]


def eval_cummulant(model, lambdas, data_loader, device):
    losses = get_loss(device, model, data_loader)
    return np.array(
        [
            (
                torch.logsumexp(- lamb * losses, 0)
                - torch.log(torch.tensor(losses.shape[0], device=device))
                + torch.mean(lamb * losses)
            )
            .detach()
            .cpu()
            .numpy()
            for lamb in lambdas
        ]
    )


def inverse_rate_function(model, lambdas, rate_vals):
    jensen_vals = eval_cummulant(model, lambdas)

    return np.array([np.min((jensen_vals + rate) / lambdas) for rate in rate_vals])
