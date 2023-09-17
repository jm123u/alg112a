#include <iostream>

// 定义一个函数来计算斐波那契数列的第n项
long long fibonacci(int n) {
    // 处理输入小于等于0的情况
    if (n <= 0) {
        return 0;
    } 
    // 处理输入为1的情况
    else if (n == 1) {
        return 1;
    }

    // 初始化前两项
    long long prev = 0;
    long long current = 1;
    long long next;

    // 使用循环计算第2到第n项
    for (int i = 2; i <= n; i++) {
        // 计算下一项并更新prev和current
        next = prev + current;
        prev = current;
        current = next;
    }

    // 返回第n项的值
    return current;
}

int main() {
    // 提示用户输入一个非负整数
    int n;
    std::cout << "Enter a non-negative integer: ";
    std::cin >> n;

    // 检查输入是否合法
    if (n < 0) {
        std::cout << "Invalid input. Please enter a non-negative integer." << std::endl;
        return 1; // 返回非零值表示程序出错
    }

    // 调用fibonacci函数计算第n项的值
    long long result = fibonacci(n);

    // 输出结果
    std::cout << "The " << n << "-th Fibonacci number is " << result << std::endl;

    return 0; // 返回零表示程序成功执行
}
