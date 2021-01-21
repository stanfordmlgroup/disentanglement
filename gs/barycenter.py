import torch
import numpy as np
import copy

# Device # this is a hack
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')

def geometricMean(T, weights=None, term=0):
    log_ = torch.log(T) - term
    if weights is None:
        mean_ = torch.mean(log_, 1)
    else:
        mean_ = torch.mm(weights, log_.t()).squeeze()
    return torch.exp(mean_)


def log_sum_exp(value, dim):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    m = m.squeeze(dim)

    sumexp = torch.sum(torch.exp(value0), dim=dim, keepdim=False)

    result = m + torch.log(sumexp)

    return result


def isnan(x):
    if x is None:
        return False
    else:
        return (x != x).any()# or (torch.abs(x) == float('inf')).any()


def initialize(C, P, epsilon, stabilization):

    if stabilization == 'original':
        V = torch.ones(P.size()).double().to(device)  # d X N
        K = torch.exp(-C / epsilon)  # d X d

        U = P / torch.mm(K, V)  # d X N
        q_all = torch.mm(K.t(), U)  # d X N
        q = geometricMean(q_all)  # d
        V = q.unsqueeze(1) / q_all  # d X N

        alpha = None
        beta = None

    elif stabilization == 'log-domain':
        alpha = torch.zeros(P.size()).double().to(device)  # d X N
        beta = torch.zeros(P.size()).double().to(device)  # d X N

        U = None
        V = None

    elif stabilization == 'log-stabilized':
        alpha = torch.zeros(P.size()).double().to(device)  # d X N
        beta = torch.zeros(P.size()).double().to(device)  # d X N
        V = torch.ones(P.size()).double().to(device)  # d X N
        K = torch.exp(-C / epsilon)  # d X d

        U = P / torch.mm(K, V)  # d X N
        q_all = torch.mm(K.t(), U)  # d X N
        q = geometricMean(q_all)  # d
        V = q.unsqueeze(1) / q_all  # d X N
    else:
        raise Exception('initialize: stabilization option is unknown')

    values = {}

    values['alpha'] = alpha
    values['beta'] = beta
    values['U'] = U
    values['V'] = V
    values['absorbed'] = False

    return values


def get_K(C, epsilon, values, stabilization):

    if stabilization == 'original':
        K = torch.exp(-C / epsilon)  # d X d

    elif stabilization == 'log-domain':
        d = values['alpha'].size(0)
        N = values['alpha'].size(1)
        K = C.unsqueeze(2).expand(d, d, N)  # d X d X N

    elif stabilization == 'log-stabilized':
        K = torch.exp((-C.unsqueeze(2) + values['alpha'].unsqueeze(1) + values['beta'].unsqueeze(0)) / epsilon)  # d X d X N
    else:
        raise Exception('get_K: stabilization option is unknown')

    return K


def update_epsilon(iter, epsilon, min_epsilon):
    if iter % 10 == 0:
        epsilon = np.max([epsilon/2, min_epsilon])
    return epsilon


def update_U(K, P, values, epsilon, stabilization, lam):
    # U = P / K*V

    if stabilization == 'original':
        # K: d X d
        U = P / torch.mm(K, values['V'])  # d X N

        # unbalanced case
        addTerm = 0
        if lam > 0:
            # U = U.pow(lam/(lam+epsilon))
            U = torch.exp(lam / (lam + epsilon) * torch.log(U + addTerm))

        values['U'] = U.to(device)

    elif stabilization == 'log-domain':
        # K = C in this case; d X d X N

        #prepare
        beta = values['beta'].unsqueeze(0).to(device)  #  1 X d X N

        #do the update
        M = (-K + beta) / epsilon  # d X d X N
        alpha = epsilon * (torch.log(P) - log_sum_exp(M, dim=1))  # d X N

        values['alpha'] = alpha.to(device)
        # print('alpha = ', alpha)

    elif stabilization == 'log-stabilized':
        # K: d X d X N
        # prepare
        K = K.permute(2, 0, 1)  # N X d X d
        V = values['V'].t().unsqueeze(2)  # N X d X 1

        # do the udpate
        U = P / (torch.bmm(K, V).squeeze().t() + 1e-100)  # d X N

        values['U'] = U

    else:
        raise Exception('update_U: stabilization option is unknown')

    return values


def update_V(K, q, q_all, values, epsilon, stabilization, lam):
    # V = q / K^T*U  or  V = q / q_all

    if stabilization == 'original':
        # prepare
        q = q.unsqueeze(1)  # d X 1

        # do the udpate
        V = q / q_all  # d X N

        # unbalanced case
        addTerm = 0
        if lam > 0:
            # V = V.pow(lam/(lam+epsilon))
            V = torch.exp(lam / (lam + epsilon) * torch.log(V + addTerm))

        values['V'] = V.to(device)

    elif stabilization == 'log-domain':
        # in this case:
        # K = C  dim = d X d X N
        # q = log_q  dim = d

        #prepare
        alpha = values['alpha'].unsqueeze(1).to(device)  # d X 1 X N
        q = q.unsqueeze(1)  # d X 1

        #do the update
        M = (-K + alpha) / epsilon  # d X d X N
        beta = epsilon * (q - log_sum_exp(M, dim=0))  # d X N

        values['beta'] = beta.to(device)

    elif stabilization == 'log-stabilized':
        # prepare
        q = q.unsqueeze(1)  # d X 1

        # do the udpate
        V = q / q_all  # d X N
        values['V'] = V
    else:
        raise Exception('update_V: stabilization option is unknown')

    return values


def update_q(K, values, epsilon, weights, stabilization, lam):
    # q = geometric_mean(K^T * U)

    if stabilization == 'original':
        # K: d X d
        q_all = torch.mm(K.t(), values['U'])  # d X N

        # unbalanced case
        addTerm = 0
        if lam > 0:
            tmp = torch.exp(epsilon / (epsilon + lam) * torch.log(q_all + addTerm)).to(device)  # d X N
            q = torch.mm(weights, tmp.t()).squeeze()
            q = q.pow((epsilon + lam) / epsilon)
            # q = q / torch.sum(q)
        else:
            q = geometricMean(q_all, weights)  # d

    elif stabilization == 'log-domain':
        # in this case:
        # K = C, dim = d X d X N
        # weights, dim = 1 X N

        # prepare
        alpha = values['alpha'].unsqueeze(1).to(device)

        # do the udpate
        M = (-K + alpha) / epsilon  # d X d X N
        log_q_all = log_sum_exp(M, dim=0)  # d X N
        log_q = torch.mm(weights, log_q_all.t()).squeeze()  # d

        q = log_q
        q_all = None

    elif stabilization == 'log-stabilized':
        # prepare
        K = K.permute(2, 1, 0)  # N X d X d (transpose)
        U = values['U'].t().unsqueeze(2)  # N X d X 1

        # do the update
        q_all = torch.bmm(K, U).squeeze().t()  # d X N
        q = geometricMean(q_all, weights, values['beta']/epsilon)  # d

    else:
        raise Exception('update_q: stabilization option is unknown')

    return q, q_all


def accelerate_U(currValues, prevValues, stabilization, tau):

    if stabilization == 'original':
        addTerm = 0
        U = torch.exp((1+tau) * torch.log(currValues['U']+addTerm) - tau * torch.log(prevValues['U']+addTerm))
        currValues['U'] = U

    # U = exp(alpha/epsilon)
    elif stabilization == 'log-domain':
        alpha = (1+tau)*currValues['alpha'] - tau*prevValues['alpha']
        currValues['alpha'] = alpha

    elif stabilization == 'log-stabilized':
        if not currValues['absorbed'] and not prevValues['absorbed']:
            addTerm = 0
            U = torch.exp((1+tau) * torch.log(currValues['U']+addTerm) - tau * torch.log(prevValues['U']+addTerm))
            currValues['U'] = U

    else:
        raise Exception('accelerate_U: stabilization option is unknown')

    return currValues


def accelerate_V(currValues, prevValues, stabilization, tau):

    if stabilization == 'original':
        addTerm = 0
        V = torch.exp((1+tau) * torch.log(currValues['V']+addTerm) - tau * torch.log(prevValues['V']+addTerm))
        currValues['V'] = V

    # V = exp(beta/epsilon)
    elif stabilization == 'log-domain':
        beta = (1+tau)*currValues['beta'] - tau*prevValues['beta']
        currValues['beta'] = beta

    elif stabilization == 'log-stabilized':
        if not currValues['absorbed'] and not prevValues['absorbed']:
            addTerm = 0
            V = torch.exp((1+tau) * torch.log(currValues['V']+addTerm) - tau * torch.log(prevValues['V']+addTerm))
            currValues['V'] = V

    else:
        raise Exception('accelerate_V: stabilization option is unknown')

    return currValues


def absorb_UV(currValues, epsilon):

    currValues['alpha'] = currValues['alpha'] + (epsilon * torch.log(currValues['U'])).to(device)
    currValues['beta'] = currValues['beta'] + (epsilon * torch.log(currValues['V'])).to(device)

    currValues['U'] = torch.ones(currValues['U'].size()).double().to(device)  # d X N
    currValues['V'] = torch.ones(currValues['V'].size()).double().to(device)  # d X N

    currValues['absorbed'] = True

    return currValues


def  computeLoss(C, K, currValues, epsilon, stabilization):

    def loss(gamma, C, epsilon):
        wasserstein = torch.sum(gamma * C)
        neg_entropy = torch.sum(epsilon * (gamma * torch.log(gamma + 1e-100) - gamma))
        loss = wasserstein + neg_entropy

        return wasserstein, neg_entropy, loss

    wasserstein = 0
    neg_entropy = 0
    lo = 0

    if stabilization == 'original':
        # K: d X d
        N = currValues['U'].size(1)

        for i in range(N):
            gamma = torch.mm(torch.mm(torch.diag(currValues['U'][:, i]), K), torch.diag(currValues['V'][:, i]))
            w, ne, l = loss(gamma, C, epsilon)
            wasserstein += w
            neg_entropy += ne
            lo += l

    elif stabilization == 'log-domain':
        #in this case K = C; K: d X d X N

        M = (-K + currValues['alpha'].unsqueeze(1).to(device) + currValues['beta'].unsqueeze(0).to(device)) / epsilon  # d X d X N
        gamma = torch.exp(M)

        N = currValues['alpha'].size(1)
        for i in range(N):
            w, ne, l = loss(gamma[:, :, i], C, epsilon)
            wasserstein += w
            neg_entropy += ne
            lo += l

    elif stabilization == 'log-stabilized':
        # K: d X d X N

        N = currValues['U'].size(1)
        for i in range(N):
            gamma = torch.mm(torch.mm(torch.diag(currValues['U'][:, i]), K[:, :, i]), torch.diag(currValues['V'][:, i]))
            w, ne, l = loss(gamma, C, epsilon)
            wasserstein += w
            neg_entropy += ne
            lo += l
    else:
        raise Exception('computeLoss: stabilization option is unknown')

    return wasserstein, neg_entropy, lo


def prepare_return(q, stabilization):

    if stabilization == 'original' or stabilization == 'log-stabilized':
        pass
    elif stabilization == 'log-domain':
        q = torch.exp(q)
    else:
        raise Exception('prepare_return: stabilization option is unknown')

    return q


def compute(P, C, weights=None, epsilon=1e-2, min_epsilon=1e-3, reduce_epsilon=0, lambda_unbalanced=-1, stabilization="original", stab_threshold=100, tau_acceleration=0, num_iterations=100, stop_threshold=1e-4, logging=0, verbose=False):
    # P: d X N - softmaxes
    # C: d X d - cost/distance (not a similarity) matrix
    # weights: 1 X N

    d = P.size(0)  # dimension of each model
    N = P.size(1)  # number of models to ensemble

    if weights is None:
        weights = torch.ones(1, N).double().to(device) / N # uniform weights

    currValues = initialize(C, P, epsilon, stabilization)

    q_diff = np.inf
    q_sum_diff = np.inf
    iter = 0
    logs = {'wass': [], 'neg_entropy': [], 'loss': [], 'epsilon':[], 'q_sum': [], 'q_diff': []}

    last_q = torch.zeros(d).double().to(device)

    while (q_diff > stop_threshold or q_sum_diff > stop_threshold) and iter < num_iterations:

        iter = iter + 1

        prevValues = copy.deepcopy(currValues)

        if reduce_epsilon:
            epsilon = update_epsilon(iter, epsilon, min_epsilon)

        K = get_K(C, epsilon, currValues, stabilization)
        # =================== update U ====================
        currValues = update_U(K, P, currValues, epsilon, stabilization, lambda_unbalanced)

        if tau_acceleration > 0:
            currValues = accelerate_U(currValues, prevValues, stabilization, tau_acceleration)

        if isnan(currValues['U']) or isnan(currValues['alpha']):
            #print('NaN in U update. Iteration =', iter)
            break

        # =================== update q ====================
        q, q_all = update_q(K, currValues, epsilon, weights, stabilization, lambda_unbalanced)

        if isnan(q):
            #print('NaN in q update. Iteration =', iter)
            break

        # =================== update V ====================
        currValues = update_V(K, q, q_all, currValues, epsilon, stabilization, lambda_unbalanced)

        if tau_acceleration > 0:
            currValues = accelerate_V(currValues, prevValues, stabilization, tau_acceleration)

        if isnan(currValues['V']) or isnan(currValues['beta']) :
            #print('NaN in V update. Iteration =', iter)
            break

        # absorption step for log-stabilized algorithm
        if stabilization == 'log-stabilized':
            if currValues['U'].abs().max() > stab_threshold or currValues['V'].abs().max() > stab_threshold:
                currValues = absorb_UV(currValues, epsilon)
            else:
                currValues['absorbed'] = False

        # exit condition
        if stabilization == 'log-domain':
            q_diff = torch.norm(torch.exp(q) - torch.exp(last_q))
            q_sum_diff = torch.abs(1.0 - torch.sum(torch.exp(q)))
        else:
            q_diff = torch.norm(q - last_q)
            q_sum_diff = torch.abs(1.0 - torch.sum(q))

        # save last good value in case next iteration fails
        last_q = q

        # =================== logging =====================
        if logging:
            wasserstein, neg_entropy, loss = computeLoss(C, K, currValues, epsilon, stabilization)
            logs['wass'].append(wasserstein)
            logs['neg_entropy'].append(neg_entropy)
            logs['loss'].append(loss)
            logs['epsilon'].append(epsilon)
            logs['q_diff'].append(q_diff)
            logs['q_sum'].append(1.0 - q_sum_diff)
            
            if verbose:
                print("# {} : epsilon= {} wass= {} neg_H= {} loss= {} q_sum= {} q_diff={}".format(iter, epsilon, wasserstein, neg_entropy, loss, logs['q_sum'][-1], q_diff))

    q = prepare_return(last_q, stabilization)

    return q, logs


def print_logs(logs):
    counter = 1
    for (wass, neg_H, loss, epsilon, q_sum, q_diff) in zip(logs['wass'], logs['neg_entropy'], logs['loss'], logs['epsilon'], logs['q_sum'], logs['q_diff']):
        print("# {} : epsilon= {} wass= {} neg_H= {} loss= {} q_sum= {} q_diff={}".format(counter, epsilon, wass, neg_H, loss, q_sum, q_diff))
        counter += 1
    return
