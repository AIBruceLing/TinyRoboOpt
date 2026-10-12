#include <iostream>
#include <cmath>
#include <Eigen/Dense>

// 高斯牛顿法：主要针对非线性最小二乘问题，利用残差函数的雅可比矩阵近似 Hessian 矩阵
// 纯高斯牛顿法就是每次步长为1，在某些情况下可能会出现 Hessian 矩阵不正定的情况，导致搜索方向不是下降方向，甚至可能会发散。
// 高斯牛顿法 + 回溯线搜索 (Armijo 准则) 实现 (阻尼高斯牛顿法)



// 高斯-牛顿的 Hessian 矩阵 H_GN = 2 * J^T * J 保证 是半正定的，搜索方向 d = -H_GN^{-1} * grad 是下降方向 
// 不能保证是严格正定的，所以

// ============================================================
// 1. 残差函数: r(x) = [ r1, r2 ]^T
//    f(x) = r1^2 + r2^2 = 100*(y - x^2)^2 + (1 - x)^2
// ============================================================
Eigen::Vector2d rosenbrock_residuals(const Eigen::Vector2d& x) {
    double r1 = 10.0 * (x[1] - x[0] * x[0]);
    double r2 = 1.0 - x[0];
    return Eigen::Vector2d(r1, r2);
}

// 目标函数: 残差平方和
double rosenbrock_cost(const Eigen::Vector2d& x) {
    Eigen::Vector2d r = rosenbrock_residuals(x);
    return r.squaredNorm();
}

// ============================================================
// 2. 残差的一阶雅可比矩阵 J (2x2 矩阵)
//    J = [ dr1/dx,  dr1/dy ] = [ -20*x,  10 ]
//        [ dr2/dx,  dr2/dy ]   [   -1 ,   0 ]
// ============================================================
Eigen::Matrix2d rosenbrock_jacobian(const Eigen::Vector2d& x) {
    Eigen::Matrix2d J;
    J(0, 0) = -20.0 * x[0];
    J(0, 1) = 10.0;
    J(1, 0) = -1.0;
    J(1, 1) = 0.0;
    return J;
}

// ============================================================
// 3. 回溯线搜索 (Armijo 准则)
// 在当前点 x 沿着方向 d 搜索步长 alpha，使得 f(x + alpha * d) <= f(x) + c1 * alpha * (grad^T * d)
// 在搜索方向确定的前提下，使用 Armijo 准则来选择合适的步长 alpha，确保每次迭代都能有效下降目标函数值
// ============================================================
double backtracking_line_search(const Eigen::Vector2d& x,     // 当前点
                                const Eigen::Vector2d& d,     // 搜索方向
                                const Eigen::Vector2d& grad,  // 当前梯度
                                double alpha_init = 1.0,      // 初始步长
                                double rho = 0.5,             // 步长缩减因子
                                double c1 = 1e-4)             // Armijo 准则参数
{
    double alpha = alpha_init;
    double current_cost = rosenbrock_cost(x);

    // 初始斜率: phi'(0) = grad^T * d
    double initial_slope = grad.dot(d);

    if (initial_slope >= 0.0) {
        std::cerr << "Warning: direction is not a descent direction!\n";
        return 0.0;
    }

    // Armijo 充分下降条件
    while (rosenbrock_cost(x + alpha * d) > current_cost + c1 * alpha * initial_slope) {
        alpha *= rho;
        if (alpha < 1e-8) {
            std::cerr << "Step size too small!\n";
            break;
        }
    }
    return alpha;
}

// ============================================================
// 4. 主循环
// ============================================================
int main() {
    std::cout << "==========================================\n";
    std::cout << "      TinyRoboOpt: Gauss-Newton Method    \n";
    std::cout << "==========================================\n";

    // 初始位置
    Eigen::Vector2d x(5.0, 5.0);
    std::cout << "Start point: (" << x[0] << ", " << x[1] << "), Initial Cost: " << rosenbrock_cost(x) << "\n\n";

    for (int iter = 0; iter < 20; ++iter) {
        Eigen::Vector2d r = rosenbrock_residuals(x); // 残差向量 r(x) , 2x1列向量
        Eigen::Matrix2d J = rosenbrock_jacobian(x);  // 计算残差的雅可比矩阵 J(x) , 2x2矩阵（注意这个是残差向量的雅可比矩阵，而不是目标函数的雅可比矩阵）

        // 1. 计算原函数（目标函数）一阶梯度: grad = 2 * J^T * r
        Eigen::Vector2d grad = 2.0 * J.transpose() * r;

        // 停机准则
        if (grad.norm() < 1e-6) {
            std::cout << "\n>>> 算法在第 " << iter << " 次迭代成功收敛到极小值点！<<<\n";
            break;
        }

        // 2. 高斯-牛顿近似 Hessian: H_GN = 2 * J^T * J
        // 正规方程: H_GN * d = -grad  等价于 (J^T * J) * d = -J^T * r
        Eigen::Matrix2d H_gn = 2.0 * J.transpose() * J;

        // 3. 直接求解高斯-牛顿方向 (因为 J^T * J 天然严格正定，用 llt 极快求解)
        Eigen::Vector2d d = H_gn.llt().solve(-grad);// 搜索方向 d = -H_GN^(-1) * grad

        // 4. 线搜索确定步长
        // d = - H(-1) * grad 
        // grad = 2 * J^T * r
        double alpha = backtracking_line_search(x, d, grad, 1.0, 0.5, 1e-4); // Armijo线搜索确定步长
        x = x + alpha * d; // 状态更新

        std::cout << "Iter " << iter + 1 
                  << " | Step alpha: " << alpha 
                  << " | Current Pose: (" << x[0] << ", " << x[1] << ")"
                  << " | Cost: " << rosenbrock_cost(x) << "\n";
    }

    std::cout << "\n最终坐标: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "最终代价值: " << rosenbrock_cost(x) << "\n";

    return 0;
}