from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf,train_features,train_label_2,scoring='accuracy')

scores.mean()

# Cross Validation Predict

from sklearn.model_selection import cross_val_predict
scores_pred=cross_val_predict(clf,train_features,train_label_2,cv=3)
scores_pred

# Cross Validation(give theresold Values)

scores_pred2 = cross_val_predict(clf,train_features,train_label_2,cv=3,method="decision_function")

scores_pred2

