#include <iostream>
#include <cmath>
#include <Eigen/Dense>

// ============================================================
// 1. 目标函数: Rosenbrock 香蕉函数
//    f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
//    全局极小值在 (1, 1), 最小值为 0
// ============================================================
double rosenbrock(const Eigen::Vector2d& x) {
    double t1 = x[1] - x[0] * x[0];
    double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
}

// ============================================================
// 2. 一阶导数: 解析梯度 (2x1 列向量)
// ============================================================
Eigen::Vector2d rosenbrock_grad(const Eigen::Vector2d& x) {
    double t1 = x[1] - x[0] * x[0];
    double df_dx = -400.0 * t1 * x[0] - 2.0 * (1.0 - x[0]);
    double df_dy = 200.0 * t1;
    return Eigen::Vector2d(df_dx, df_dy);
}

// ============================================================
// 3. 二阶导数: 解析 Hessian 矩阵 (2x2 对称矩阵)
//    H = [ 1200*x^2 - 400*y + 2,   -400*x ]
//        [         -400*x        ,    200   ]
// ============================================================
Eigen::Matrix2d rosenbrock_hessian(const Eigen::Vector2d& x) {
    Eigen::Matrix2d H;
    H(0, 0) = 1200.0 * x[0] * x[0] - 400.0 * x[1] + 2.0;
    H(0, 1) = -400.0 * x[0];
    H(1, 0) = -400.0 * x[0];
    H(1, 1) = 200.0;
    return H;
}

// ============================================================
// 4. 回溯线搜索 (Armijo 准则)
// ============================================================
double backtracking_line_search(const Eigen::Vector2d& x,
                                const Eigen::Vector2d& d,
                                double alpha_init = 1.0,
                                double rho = 0.5,
                                double c1 = 1e-4) 
{
    double alpha = alpha_init;
    double current_cost = rosenbrock(x);

    // 初始斜率 (方向导数): phi'(0) = grad^T * d
    double initial_slope = rosenbrock_grad(x).dot(d);

    if (initial_slope >= 0.0) {
        std::cerr << "Warning: direction is not a descent direction!\n";
        return 0.0;
    }

    // Armijo 充分下降条件
    while (rosenbrock(x + alpha * d) > current_cost + c1 * alpha * initial_slope) {
        alpha *= rho;
        if (alpha < 1e-8) {
            std::cerr << "Step size too small!\n";
            break;
        }
    }
    return alpha;
}

// ============================================================
// 5. 核心模块：LM 阻尼正则化求解牛顿方向
//    求解: (H + mu * I) * d = -grad
// ============================================================
Eigen::Vector2d compute_lm_newton_direction(const Eigen::Matrix2d& H, 
                                            const Eigen::Vector2d& grad, 
                                            double& final_mu) 
{
    double mu = 0.0; // 默认先尝试纯牛顿法 (无阻尼)
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    Eigen::Vector2d d;

    while (true) {
        Eigen::Matrix2d H_reg = H + mu * I; // LM修正后的hessian矩阵

        // 利用 Cholesky 分解 (LLT) 检查矩阵是否严格正定
        Eigen::LLT<Eigen::Matrix2d> llt(H_reg);

        if (llt.info() == Eigen::Success) {
            // 分解成功说明 H_reg 对称正定，解方程 (H + mu*I) * d = -grad
            d = llt.solve(-grad); // d = -(H + mu*I)^(-1) * grad

            // 再次确保方向是严格下坡方向 (grad^T * d < 0)
            if (grad.dot(d) < 0.0) {
                final_mu = mu;
                return d; // 成功找到正则化牛顿方向！
            }
        }

        // 若不正定或方向不是严格下坡，自适应放大阻尼因子 mu (LM 机制)
        mu = (mu == 0.0) ? 1e-3 : mu * 10.0;

        if (mu > 1e9) {
            std::cerr << "Warning: Regularization parameter mu reached upper limit!\n";
            final_mu = mu;
            return -grad; // 极端防御性保底
        }
    }
}

// ============================================================
// 6. 主循环
// ============================================================
int main() {
    std::cout << "==========================================\n";
    std::cout << "      TinyRoboOpt: Newton-LM Method       \n";
    std::cout << "==========================================\n";

    // 初始位置 (可尝试 (5, 5)，也可尝试极端非正定区域如 (0, 1))
    Eigen::Vector2d x(5.0, 5.0);
    std::cout << "Start point: (" << x[0] << ", " << x[1] << "), Initial Cost: " << rosenbrock(x) << "\n\n";

    for (int iter = 0; iter < 50; ++iter) {
        Eigen::Vector2d grad = rosenbrock_grad(x);

        // 停机准则
        if (grad.norm() < 1e-6) {
            std::cout << "\n>>> 算法在第 " << iter << " 次迭代成功收敛到极小值点！<<<\n";
            break;
        }

        Eigen::Matrix2d H = rosenbrock_hessian(x);

        // ★★★ 计算带 LM 修正的牛顿方向 ★★★
        double mu = 0.0;
        Eigen::Vector2d d = compute_lm_newton_direction(H, grad, mu);

        // 线搜索确定步长 (牛顿法默认尝试完整步长 alpha_init = 1.0)
        double alpha = backtracking_line_search(x, d, 1.0, 0.5, 1e-4);
        x = x + alpha * d; // 状态更新

        std::cout << "Iter " << iter + 1 
                  << " | mu: " << mu
                  << " | Step alpha: " << alpha 
                  << " | Current Pose: (" << x[0] << ", " << x[1] << ")"
                  << " | Cost: " << rosenbrock(x) << "\n";
    }

    std::cout << "\n最终坐标: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "最终代价值: " << rosenbrock(x) << "\n";

    return 0;
}