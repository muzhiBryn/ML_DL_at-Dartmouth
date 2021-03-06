from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply

    # %%% DO NOT EDIT ABOVE


    # let y be the output of fully connected(X, W, B)
    y = full_connected(X, W, B)
    # z has to be a scaler and a function of y, mean() will do it
    z = y.mean()
    # dzdy used in equation (6) to compute numerical gradient
    dzdy, = torch.autograd.grad(z, y)  #dzdy here is 48*72

    # Then you start doing eq(1)(2)...(6)
    t = X.size(0) #number of samples - 48
    n = X.size(1) #number of features in X .. 2
    m = W.size(1) # number of units in output ... 72

    # numerical_dzdx
    numerical_dzdx = torch.zeros(t,n)
    for i in range(n):
        mask = torch.zeros(n)
        mask[i] = 1
        numerical_dzdx[:, i] = (dzdy * ((full_connected(X + DELTA * mask, W, B) - full_connected(X - DELTA * mask, W, B)) / (2 * DELTA))).sum(dim=1)

    # numerical_dzdw
    numerical_dzdw = torch.zeros(n, m)
    for i in range(n):
        for j in range(m):
            mask = torch.zeros(n, m)
            mask[i, j] = 1
            numerical_dzdw[i, j] = (dzdy * ((full_connected(X, W + DELTA * mask, B) - full_connected(X, W - DELTA * mask, B)) / (2 * DELTA))).sum()

    # numerical_dzdb
    numerical_dzdb = torch.zeros(m)
    for i in range(m):
        mask = torch.zeros(m)
        mask[i] = 1
        numerical_dzdb[i] = (dzdy * (
                    (full_connected(X, W, B + DELTA * mask) - full_connected(X, W, B - DELTA * mask)) / (
                        2 * DELTA))).sum()

    z.backward()
    # analytical gradients
    dzdx, dzdw, dzdb = X.grad, W.grad, B.grad

    # Finally check if analytical and numerical gradients match

# (numerical_dzdx.size(), dzdx.size())
#     err_w = print( dzdw.size())
#     err_b = print(dzdb.size())
    def helper(numerical_value, value):
        err = (numerical_value - value).abs().max()
        if err > TOL:
            return False, err
        else:
            return True, err

    dzdx_correct, dzdx_err = helper(numerical_dzdx, dzdx)
    dzdw_correct, dzdw_err = helper(numerical_dzdw, dzdw)
    dzdb_correct, dzdb_err = helper(numerical_dzdb, dzdb)

    is_correct = dzdx_correct and dzdw_correct and dzdb_correct and torch.autograd.gradcheck(full_connected, (X, W, B), eps=DELTA, atol=TOL)

    err = {'dzdx': dzdx_err, 'dzdw': dzdw_err, 'dzdb': dzdw_err}
    torch.save([is_correct, err], 'fully_connected_test_results.pt')

    return is_correct, err



if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)
