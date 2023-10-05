# get Accuracy in One-hot-encoding

predictions = [[1, 0, 0, 0], [0, 0, 1, 0],
               [0, 0, 1, 0], [1, 0, 0, 0],
               [1, 0, 0, 0], [0, 0, 0, 1]]
labels = [0, 1, 2, 1, 0, 3]

n_label = len(labels)
n_class = 0

for label in labels:
    if label > n_class:
        n_class = label

n_class += 1

one_hot_mat = list()

for label in labels:
    one_hot_vec = list()
    for _ in range(n_class):
        one_hot_vec.append(0)
    one_hot_vec[label] = 1

    one_hot_mat.append(one_hot_vec)

label_correct_cnt = 0

for dim1_idx in range(len(predictions)):
    correct_cnt = 0
    for dim2_idx in range(len(predictions[0])):
        if predictions[dim1_idx][dim2_idx] == one_hot_mat[dim1_idx][dim2_idx]:
            correct_cnt += 1
    if correct_cnt == len(predictions[0]):
        label_correct_cnt += 1

accuracy = label_correct_cnt / n_label

print("predictions: ", predictions)
print("labels: ", one_hot_mat)
print("accuracy: ", accuracy)
