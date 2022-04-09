from sklearn.metrics import precision_recall_curve
# precision = precision_recall_curve(train_label_2)
# recall = precision_recall_curve(scores_pred)
# theresold = precision_recall_curve(scores_pred2)
precision,recalls,theresolds =precision_recall_curve(train_label_2,scores_pred2)


# precision.shape
# recalls.shape
theresolds.shape

