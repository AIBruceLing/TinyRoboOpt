#include <iostream>
#include <cmath>
#include <Eigen/Dense>

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
// ============================================================
double backtracking_line_search(const Eigen::Vector2d& x,
                                const Eigen::Vector2d& d,
                                const Eigen::Vector2d& grad,
                                double alpha_init = 1.0,
                                double rho = 0.5,
                                double c1 = 1e-4) 
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
        Eigen::Vector2d r = rosenbrock_residuals(x);
        Eigen::Matrix2d J = rosenbrock_jacobian(x);

        // 1. 计算一阶梯度: grad = 2 * J^T * r
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
        Eigen::Vector2d d = H_gn.llt().solve(-grad);

        // 4. 线搜索确定步长
        double alpha = backtracking_line_search(x, d, grad, 1.0, 0.5, 1e-4);
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