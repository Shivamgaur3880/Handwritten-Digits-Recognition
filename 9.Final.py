plt.plot(theresolds,precision[:-1],"b--",label="precisions")  # precision and recall me 1 value jyada hain
plt.plot(theresolds,recalls[:-1],"g-",label="recalls")
plt.xlabel("theresold")
plt.legend(loc="upper left")   #location
# plt.ylim([0,1])
plt.show()

# if precision increse then recalls decrease
# if want more positive prediction then there is Precision

# if want more negative prediction then there is Recall