import numpy as np


def adam(grad_fn, x_init, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, num_iters=1000):
    x = x_init
    m = 0
    v = 0
    t = 0

    for i in range(num_iters):
        grad = grad_fn(x)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** (t+1))
        v_hat = v / (1 - beta2 ** (t+1))

        x -= alpha * m_hat / (np.sqrt(v_hat) + eps)

        t += 1

    return x
