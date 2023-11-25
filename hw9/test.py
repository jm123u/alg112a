import json

def editDistanceRecursive(a, b):
    memo = {}

    def helper(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        if i == 0:
            result = j
        elif j == 0:
            result = i
        else:
            cost = 0 if a[i - 1] == b[j - 1] else 1

            result = min(
                helper(i-1, j) + 1,
                helper(i, j-1) + 1,
                helper(i-1, j-1) + cost
            )

        memo[(i, j)] = result
        return result

    distance = helper(len(a), len(b))
    return {'d': distance, 'm': memo}

def dump(m):
    for key, value in m.items():
        print(json.dumps({str(key): value}))

a = 'ATGCAATCCC'
b = 'ATGATCCG'

result = editDistanceRecursive(a, b)
print(f'editDistanceRecursive({a}, {b}) = {result["d"]}')
print('====m======\n')
dump(result['m'])