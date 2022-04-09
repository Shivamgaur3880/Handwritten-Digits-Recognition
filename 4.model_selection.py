from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

clf.fit(train_features,train_label_2)

clf.predict([some_digit])

