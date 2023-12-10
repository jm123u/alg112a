# 簡單 示例 由chatgpt生成
def is_satisfiable(variables, clauses, assignment):
    for clause in clauses:
        satisfied = False
        for literal in clause:
            variable, is_negated = abs(literal), literal < 0
            if (variable in assignment and assignment[variable] == (not is_negated)):
                satisfied = True
                break
        if not satisfied:
            return False
    return True

def verify_3sat():
    # 3SAT问题的例子
    variables = {1, 2, 3}
    clauses = [(1, 2, -3), (-1, 2, 3), (-1, -2, -3)]

    # 假设的解
    assignment = {1: True, 2: False, 3: True}

    # 验证解是否正确
    if is_satisfiable(variables, clauses, assignment):
        print("解是正确的")
    else:
        print("解是错误的")

if __name__ == "__main__":
    verify_3sat()
