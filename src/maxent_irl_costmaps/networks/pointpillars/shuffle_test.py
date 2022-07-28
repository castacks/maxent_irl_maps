import numpy as np
import time

if __name__ == '__main__':
    N = 100
    X = np.random.randn(100000, 3)

    t1 = time.time()
    for i in range(N):
        np.random.shuffle(X)

    t2 = time.time()

    print('shuffle time = {:.4f}s'.format((t2-t1)/N))
