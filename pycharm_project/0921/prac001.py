# get Accuracy

predictions = [0, 1, 0, 2, 1, 2, 0]
labels = [1, 1, 0, 0, 1, 2, 1]
n_correct = 0

for pred_idx in range(len(predictions)):
    if predictions[pred_idx] == labels[pred_idx]:
        n_correct += 1

accuracy = n_correct / len(predictions)
print("accuracy[%]: ", accuracy * 100, '%')
