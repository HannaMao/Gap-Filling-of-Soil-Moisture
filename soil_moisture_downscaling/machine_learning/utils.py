# Author: Hanzi Mao <hannamao15@gmail.com>
#
# License: BSD 3 clause


def pre_process(X_train, X_test):
    # order = np.argsort(np.random.random(self.y_train.shape))
    # self.X_train = self.X_train[order]
    # self.y_train = self.y_train[order]
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(mean, std)
    print(X_train.shape)
    print(X_test.shape)
    return X_train, X_test