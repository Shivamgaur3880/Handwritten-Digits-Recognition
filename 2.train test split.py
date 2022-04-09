train_features = x.iloc[:4000]
train_label = y.iloc[:4000]
train_label.shape

test_feature = x.iloc[4000:]
test_label = y.iloc[4000:]


# SHUFFLING
shuffle_index = np.random.permutation(4000)
train_features , train_label = train_features.iloc[shuffle_index] , train_label.iloc[shuffle_index]

