import numpy as np


# class BinaryCrossEntropy:
#     def __init__(self, y):
#         self.y = y
#
#     def get_loss(self, y_pred):
#         loss = -(self.y * np.log(y_pred) + ((1 - self.y) * np.log(1 - y_pred)))
#
#         return loss
#
#
# bce = BinaryCrossEntropy(0)
#
# print(bce.get_loss(0.9))


class BCELoss:
    def forward(self, y, pred):
        j = -(y * np.log(pred) + ((1 - y) * np.log(1 - pred)))

        return j


loss_function = BCELoss()
preds = np.arange(0.1, 1, 0.1)
print("Case.1) y = 0")
for pred in preds:
    print(loss_function.forward(0, pred))

print("\nCase.2) y = 1")
for pred in preds:
    print(loss_function.forward(1, pred))
