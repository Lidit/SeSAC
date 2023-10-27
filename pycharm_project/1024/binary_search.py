def bn_search(arr, length, target: int):
    first = 0
    last = length - 1

    while first <= last:
        mid = int((first + last) / 2)

        if target == arr[mid]:
            return mid
        else:
            if target < arr[mid]:
                last = mid - 1
            else:
                first = mid + 1

    return -1


ar = [1, 3, 5, 7, 9]
idx = bn_search(ar, len(ar), 7)

if idx == -1:
    print("탐색 실패")
else:
    print("타겟 저장 인덱스: ", idx)

idx = bn_search(ar, len(ar), 4)
if idx == -1:
    print("탐색 실패")
else:
    print("타겟 저장 인덱스: ", idx)
