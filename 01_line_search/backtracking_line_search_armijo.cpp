#include <iostream>
#include <cmath>
#include <Eigen/Dense>

// 目标函数: f(x, y) = x^2 + 2*y^2 (极小值在原点 0, 0)
double cost_func(const Eigen::Vector2d& x) {
    return x[0] * x[0] + 2.0 * x[1] * x[1];
}

// 目标函数的解析梯度（列向量）: grad = [2x, 4y]^T
Eigen::Vector2d gradient(const Eigen::Vector2d& x) {
    return Eigen::Vector2d(2.0 * x[0], 4.0 * x[1]);
}


// rosenbrock 香蕉函数 (极小值在 (1, 1), 值为 0) 
// rosenbrock函数用梯度下降法就效率很低了，需要迭代上万次，所以必须引入曲率，也就是hessian矩阵，比如牛顿法、L-BFGS
//    f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2

double rosenbrock(const Eigen::Vector2d& x) {
    double t1 = x[1] - x[0] * x[0];
    double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
}

Eigen::Vector2d rosenbrock_grad(const Eigen::Vector2d& x) {
    double t1 = x[1] - x[0] * x[0];
    double df_dx = -400.0 * t1 * x[0] - 2.0 * (1.0 - x[0]);
    double df_dy = 200.0 * t1;
    return Eigen::Vector2d(df_dx, df_dy);
}
 
//  回溯线搜索（非精确） + Armijo 条件 函数
double backtracking_line_search(const Eigen::Vector2d& x,  // 当前点
                                const Eigen::Vector2d& d,  // 搜索方向，是有尺度的
                                double alpha_init = 1.0,   // 初始步长，相当于对d的折扣系数
                                double rho = 0.5,          // 步长衰减因子
                                double c1 = 1e-4)          // Armijo 准则参数
{
    double alpha = alpha_init;
    double current_cost = rosenbrock(x);

    // 计算初始斜率（方向导数，是一个标量）
    // 梯度和搜索方向的点积是沿着搜索方向d的真实斜率，斜率 = 梯度⋅搜索方向
    // 斜率物理意义：迈出一步后函数值的变化
    double initial_slope = rosenbrock_grad(x).dot(d); // phi'(0) = grad^T * d

    if (initial_slope >= 0.0) {
        std::cerr << "Warning: direction is not a descent direction!\n";
        return 0.0;
    }

    // Armijo 准则循环: f(x + alpha * d) <= f(x) + c1 * alpha * (grad^T * d)
    while (rosenbrock(x + alpha * d) > current_cost + c1 * alpha * initial_slope) {
        alpha *= rho; // 不满足充分下降，步长打折缩短
        if (alpha < 1e-8) {
            std::cerr << "Step size too small!\n";
            break;
        }
    }
    return alpha;
}

int main() {
    std::cout << "==========================================\n";
    std::cout << "      TinyRoboOpt: Armijo Line Search     \n";
    std::cout << "==========================================\n";

    // 初始位置
    Eigen::Vector2d x(2.0, 2.5);
    std::cout << "Start point: (" << x[0] << ", " << x[1] << "), Initial Cost: " << rosenbrock(x) << "\n\n";

    // 迭代 5 次观察下坡过程
    for (int iter = 0; iter < 100000; ++iter) {
        Eigen::Vector2d grad = rosenbrock_grad(x);

        if(grad.norm() < 1e-6) {
            std::cout << "\n>>> 算法在第 " << iter << " 次迭代成功收敛到极小值点！<<<\n";
            break;
        }
        Eigen::Vector2d d = -grad; // 最速下降方向 (负梯度)

        double alpha = backtracking_line_search(x, d, 1.0, 0.5, 1e-4);
        x = x + alpha * d; // 状态更新

        std::cout << "Iter " << iter + 1 
                  << " | Step alpha: " << alpha 
                  << " | Current Pose: (" << x[0] << ", " << x[1] << ")"
                  << " | Cost: " << rosenbrock(x) << "\n";
    }

    std::cout << "\n>>> 恭喜！你的第一个最优化算法闭环已成功跑通！\n";
    return 0;
}