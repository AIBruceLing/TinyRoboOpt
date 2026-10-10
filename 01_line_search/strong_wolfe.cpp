#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <Eigen/Dense>

// ============================================================
// 1. 测试目标函数: Rosenbrock 香蕉函数 (极小值在 (1, 1), 值为 0)
//    f(x, y) = 100 * (y - x^2)^2 + (1 - x)^2
// ============================================================
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

// ============================================================
// 2. 强 Wolfe 线搜索实现 (标准 Nocedal & Wright 算法架构)
// ============================================================
class StrongWolfeLineSearch {
public:
    double c1 = 1e-4; // Armijo 充分下降参数 (通常 1e-4)
    double c2 = 0.9;  // 强曲率参数 (拟牛顿法通常设为 0.9, 共轭梯度设为 0.1)

    // 辅助函数: Zoom 阶段 (在区间 [a_lo, a_hi] 内二分收缩)
    double zoom(double a_lo, double a_hi, 
                const Eigen::Vector2d& x, const Eigen::Vector2d& d,
                double phi_0, double dphi_0,
                const std::function<double(const Eigen::Vector2d&)>& func,
                const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& grad,
                int max_zoom_iter = 10) 
    {
        for (int iter = 0; iter < max_zoom_iter; ++iter) {
            // 使用二分插值 (也可以换成二次/三次插值以加速)
            double a_j = 0.5 * (a_lo + a_hi);
            double phi_aj = func(x + a_j * d);
            double phi_alo = func(x + a_lo * d);

            // 检查 Armijo 条件
            if (phi_aj > phi_0 + c1 * a_j * dphi_0 || phi_aj >= phi_alo) {
                a_hi = a_j; // 步长过大或没改善，收缩右边界
            } else {
                double dphi_aj = grad(x + a_j * d).dot(d);

                // 检查【强曲率条件】: |phi'(a_j)| <= c2 * |phi'(0)|
                if (std::abs(dphi_aj) <= -c2 * dphi_0) {
                    return a_j; // 完美满足强 Wolfe！
                }

                if (dphi_aj * (a_hi - a_lo) >= 0.0) {
                    a_hi = a_lo;
                }
                a_lo = a_j;
            }
        }
        return 0.5 * (a_lo + a_hi);
    }

    // 核心搜索入口
    double search(const Eigen::Vector2d& x, const Eigen::Vector2d& d,
                  const std::function<double(const Eigen::Vector2d&)>& func,
                  const std::function<Eigen::Vector2d(const Eigen::Vector2d&)>& grad,
                  double a_init = 1.0, double a_max = 10.0, int max_iter = 20) 
    {
        double phi_0 = func(x);
        double dphi_0 = grad(x).dot(d); // phi'(0) = grad^T * d

        if (dphi_0 >= 0.0) {
            std::cerr << "[Error]: 不是有效的下降方向！\n";
            return 0.0;
        }

        double a_prev = 0.0;
        double a_curr = a_init;
        double phi_prev = phi_0;

        for (int i = 1; i <= max_iter; ++i) {
            double phi_curr = func(x + a_curr * d);

            // 1. 检查 Armijo 充分下降条件
            if (phi_curr > phi_0 + c1 * a_curr * dphi_0 || (i > 1 && phi_curr >= phi_prev)) {
                return zoom(a_prev, a_curr, x, d, phi_0, dphi_0, func, grad);
            }

            double dphi_curr = grad(x + a_curr * d).dot(d);

            // 2. 检查【强曲率条件】: |phi'(a)| <= c2 * |phi'(0)|
            if (std::abs(dphi_curr) <= -c2 * dphi_0) {
                return a_curr; // 步长完美！
            }

            // 3. 如果斜率变为正数 (冲过谷底开始上坡了)
            if (dphi_curr >= 0.0) {
                return zoom(a_curr, a_prev, x, d, phi_0, dphi_0, func, grad);
            }

            // 4. 还是陡坡，继续外插放大步长
            a_prev = a_curr;
            phi_prev = phi_curr;
            a_curr = std::min(2.0 * a_curr, a_max);

            if (a_curr >= a_max) {
                break;
            }
        }
        return a_curr;
    }
};

// ============================================================
// 3. 测试主函数
// ============================================================
int main() {
    std::cout << "========================================================\n";
    std::cout << "      TinyRoboOpt: Strong Wolfe Line Search Demo        \n";
    std::cout << "========================================================\n\n";

    StrongWolfeLineSearch line_search;
    
    // 从经典香蕉函数刁钻起点出发: (-1.2, 1.0)
    Eigen::Vector2d x(-1.2, 1.0);
    std::cout << "起点 x_0: (" << x[0] << ", " << x[1] << ")\n";
    std::cout << "初始代价 Cost: " << rosenbrock(x) << "\n\n";

    for (int step = 0; step < 5; ++step) {
        Eigen::Vector2d g = rosenbrock_grad(x);
        Eigen::Vector2d d = -g; // 负梯度方向

        double alpha = line_search.search(x, d, rosenbrock, rosenbrock_grad, 1.0);

        Eigen::Vector2d x_new = x + alpha * d;
        Eigen::Vector2d g_new = rosenbrock_grad(x_new);

        // 验证物理量
        Eigen::Vector2d s = x_new - x;      // 位置差
        Eigen::Vector2d y = g_new - g;      // 梯度差
        double s_dot_y = s.dot(y);          // 拟牛顿法正定性检验指标!

        double phi_0 = rosenbrock(x);
        double phi_alpha = rosenbrock(x_new);
        double dphi_0 = g.dot(d);
        double dphi_alpha = g_new.dot(d);

        std::cout << "--- 迭代 Step " << step + 1 << " ---\n";
        std::cout << "选取的步长 alpha: " << alpha << "\n";
        std::cout << "更新后位置 x: (" << x_new[0] << ", " << x_new[1] << ")\n";
        std::cout << "代价下降: " << phi_0 << " -> " << phi_alpha << "\n";
        std::cout << "Armijo 检验 (应该为真): " 
                  << (phi_alpha <= phi_0 + line_search.c1 * alpha * dphi_0 ? "PASSED" : "FAILED") << "\n";
        std::cout << "强曲率检验 |phi'| <= c2*|phi'0|: " 
                  << (std::abs(dphi_alpha) <= -line_search.c2 * dphi_0 ? "PASSED" : "FAILED") << "\n";
        std::cout << "★ 拟牛顿正定性指标 s^T * y (必须 > 0): " << s_dot_y << " (" 
                  << (s_dot_y > 0 ? "合格 ✓" : "失败 ✗") << ")\n\n";

        x = x_new;
    }

    return 0;
}