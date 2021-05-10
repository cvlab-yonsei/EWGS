from custom_modules import *
from custom_models import *
import numpy as  np
import torch
import sys
from tqdm import tqdm

__all__ = ['init_quant_model', 'update_grad_scales']

def init_quant_model(model, train_loader, device):
    for m in model.modules():
        if isinstance(m, QConv):
            m.init.data.fill_(1)
    
    iterloader = iter(train_loader)
    images, labels = next(iterloader)

    images = images.to(device)

    model.train()
    model.forward(images)
    for m in model.modules():
        if isinstance(m, QConv):
            m.init.data.fill_(0)

def update_grad_scales(model, train_loader, criterion, device, args):
    ## update scales
    if args.QActFlag:
        scaleA = []
    if args.QWeightFlag:
        scaleW = []
    for m in model.modules():
        if isinstance(m, QConv):
            m.hook_Qvalues = True
            if args.QActFlag:
                scaleA.append(0)
            if args.QWeightFlag:
                scaleW.append(0)

    model.train()
    with tqdm(total=3, file=sys.stdout) as pbar:
        for num_batches, (images, labels) in enumerate(train_loader):
            if num_batches == 3: # estimate trace using 3 batches
                break
            images = images.to(device)
            labels = labels.to(device)

            # forward with single batch
            model.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward(create_graph=True)

            # store quantized values
            if args.QWeightFlag:
                Qweight = []
            if args.QActFlag:
                Qact = []
            for m in model.modules():
                if isinstance(m, QConv):
                    if args.QWeightFlag:
                        Qweight.append(m.buff_weight)
                    if args.QActFlag:
                        Qact.append(m.buff_act)

            # update the scaling factor for activations
            if args.QActFlag:
                params = []
                grads = []
                for i in range(len(Qact)): # store variable & gradients
                    params.append(Qact[i])
                    grads.append(Qact[i].grad)

                for i in range(len(Qact)):
                    trace_hess_A = np.mean(trace(model, [params[i]], [grads[i]], device))
                    avg_trace_hess_A = trace_hess_A / params[i].view(-1).size()[0] # avg trace of hessian
                    scaleA[i] += (avg_trace_hess_A / (grads[i].std().cpu().item()*3.0))
                
            # update the scaling factor for weights
            if args.QWeightFlag:
                params = []
                grads = []
                for i in range(len(Qweight)):
                    params.append(Qweight[i])
                    grads.append(Qweight[i].grad)

                for i in range(len(Qweight)):
                    trace_hess_W = np.mean(trace(model, [params[i]], [grads[i]], device))
                    avg_trace_hess_W = trace_hess_W / params[i].view(-1).size()[0]
                    scaleW[i] += (avg_trace_hess_W / (grads[i].std().cpu().item()*3.0))
            pbar.update(1)
        
    
    if args.QActFlag:
        for i in range(len(scaleA)):
            scaleA[i] /= num_batches
            scaleA[i] = np.clip(scaleA[i], 0, np.inf)
        print("\n\nscaleA\n", scaleA)
    if args.QWeightFlag:
        for i in range(len(scaleW)):
            scaleW[i] /= num_batches
            scaleW[i] = np.clip(scaleW[i], 0, np.inf)
        print("scaleW\n", scaleW)       
    print("")

    i = 0
    for m in model.modules():
        if isinstance(m, QConv):
            if args.QWeightFlag:
                m.bkwd_scaling_factorW.data.fill_(scaleW[i])
            if args.QActFlag:
                m.bkwd_scaling_factorA.data.fill_(scaleA[i])
            m.hook_Qvalues = False
            i += 1


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv

def trace(model, params, grads, device, maxIter=50, tol=1e-3):
    """
    compute the trace of hessian using Hutchinson's method
    maxIter: maximum iterations used to compute trace
    tol: the relative tolerance
    """

    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
        model.zero_grad()
        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]
        # generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        
        Hv = hessian_vector_product(grads, params, v)
        trace_vhv.append(group_product(Hv, v).cpu().item())
        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            return trace_vhv
        else:
            trace = np.mean(trace_vhv)

    return trace_vhv
