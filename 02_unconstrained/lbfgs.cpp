#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <Eigen/Dense>

// =========================================================================
// 1. 目标函数: Rosenbrock 香蕉函数 (极小值点在 (1.0, 1.0), 值为 0)
// =========================================================================
double cost_func(const Eigen::VectorXd& x) {
    double t1 = x[1] - x[0] * x[0];
    double t2 = 1.0 - x[0];
    return 100.0 * t1 * t1 + t2 * t2;
}

Eigen::VectorXd grad_func(const Eigen::VectorXd& x) {
    Eigen::VectorXd g(2);
    double t1 = x[1] - x[0] * x[0];
    g[0] = -400.0 * t1 * x[0] - 2.0 * (1.0 - x[0]);
    g[1] = 200.0 * t1;
    return g;
}

// =========================================================================
// 2. 强 Wolfe 线搜索模块 (带 Zoom 二分收缩与外插放大)
// =========================================================================
class StrongWolfeLineSearch {
public:
    double c1 = 1e-4; // 充分下降参数
    double c2 = 0.9;  // 强曲率参数

    double zoom(double a_lo, double a_hi, 
                const Eigen::VectorXd& x, const Eigen::VectorXd& d,
                double phi_0, double dphi_0, int max_zoom = 10) 
    {
        for (int iter = 0; iter < max_zoom; ++iter) {
            double a_j = 0.5 * (a_lo + a_hi);
            double phi_aj = cost_func(x + a_j * d);
            double phi_alo = cost_func(x + a_lo * d);

            if (phi_aj > phi_0 + c1 * a_j * dphi_0 || phi_aj >= phi_alo) {
                a_hi = a_j;
            } else {
                double dphi_aj = grad_func(x + a_j * d).dot(d);
                if (std::abs(dphi_aj) <= -c2 * dphi_0) {
                    return a_j; // 满足强 Wolfe 条件！
                }
                if (dphi_aj * (a_hi - a_lo) >= 0.0) {
                    a_hi = a_lo;
                }
                a_lo = a_j;
            }
        }
        return 0.5 * (a_lo + a_hi);
    }

    double search(const Eigen::VectorXd& x, const Eigen::VectorXd& d, double a_init = 1.0) {
        double phi_0 = cost_func(x);
        double dphi_0 = grad_func(x).dot(d);

        if (dphi_0 >= 0.0) return 0.0; // 不是有效下坡方向

        double a_prev = 0.0;
        double a_curr = a_init;
        double phi_prev = phi_0;
        double a_max = 50.0; // 允许向外放大探测

        for (int i = 1; i <= 25; ++i) {
            double phi_curr = cost_func(x + a_curr * d);

            if (phi_curr > phi_0 + c1 * a_curr * dphi_0 || (i > 1 && phi_curr >= phi_prev)) {
                return zoom(a_prev, a_curr, x, d, phi_0, dphi_0);
            }

            double dphi_curr = grad_func(x + a_curr * d).dot(d);
            if (std::abs(dphi_curr) <= -c2 * dphi_0) {
                return a_curr;
            }

            if (dphi_curr >= 0.0) {
                return zoom(a_curr, a_prev, x, d, phi_0, dphi_0);
            }

            // 关键：如果不满足曲率条件，强行向外加倍放大步长 (打破爬行死锁！)
            a_prev = a_curr;
            phi_prev = phi_curr;
            a_curr = std::min(2.0 * a_curr, a_max);
            if (a_curr >= a_max) break;
        }
        return a_curr;
    }
};

// =========================================================================
// 3. L-BFGS 历史曲率记忆容器
// =========================================================================
struct LBFGSHistory {
    int m = 8; // 保持最近 8 步记忆
    std::deque<Eigen::VectorXd> s_list; // 位移差
    std::deque<Eigen::VectorXd> y_list; // 梯度差
    std::deque<double> rho_list;

    void push(const Eigen::VectorXd& s, const Eigen::VectorXd& y) {
        double sy = s.dot(y);
        if (sy > 1e-10) { // 必须满足正定性
            if (s_list.size() >= static_cast<size_t>(m)) {
                s_list.pop_front();
                y_list.pop_front();
                rho_list.pop_front();
            }
            s_list.push_back(s);
            y_list.push_back(y);
            rho_list.push_back(1.0 / sy);
        }
    }
};

// =========================================================================
// 4. L-BFGS 双循环递归算法 (Two-Loop Recursion)
// =========================================================================
Eigen::VectorXd compute_lbfgs_direction(const Eigen::VectorXd& g, const LBFGSHistory& hist) {
    int k = hist.s_list.size();
    if (k == 0) return -g;

    Eigen::VectorXd q = g;
    std::vector<double> alpha(k);

    // 第一层循环：从新到旧
    for (int i = k - 1; i >= 0; --i) {
        alpha[i] = hist.rho_list[i] * hist.s_list[i].dot(q);
        q -= alpha[i] * hist.y_list[i];
    }

    // 初始海森逆缩放: gamma = (s^T * y) / ||y||^2
    double gamma = hist.s_list.back().dot(hist.y_list.back()) / hist.y_list.back().squaredNorm();
    Eigen::VectorXd r = gamma * q;

    // 第二层循环：从旧到新
    for (int i = 0; i < k; ++i) {
        double beta = hist.rho_list[i] * hist.y_list[i].dot(r);
        r += hist.s_list[i] * (alpha[i] - beta);
    }

    return -r; // 返回矫正后的牛顿下山方向
}

// =========================================================================
// 5. 优化主函数
// =========================================================================
int main() {
    std::cout << "======================================================================\n";
    std::cout << "     TinyRoboOpt: Fixed Mini-LBFGS (with Strong Wolfe Line Search)     \n";
    std::cout << "======================================================================\n";

    Eigen::VectorXd x(2);
    x << -1.2, 1.0; // 起点

    LBFGSHistory hist;
    StrongWolfeLineSearch line_search;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Iter |      Cost      |    ||Grad||    |   Step alpha   | Current x\n";
    std::cout << "----------------------------------------------------------------------\n";

    for (int iter = 0; iter < 50; ++iter) {
        Eigen::VectorXd g = grad_func(x);
        double g_norm = g.norm();
        double cost = cost_func(x);

        std::cout << std::setw(4) << iter << " | "
                  << std::setw(14) << cost << " | "
                  << std::setw(14) << g_norm << " | ";

        if (g_norm < 1e-5) {
            std::cout << "\n>>> 优化成功收敛！精准命中极小值点！\n";
            break;
        }

        // 1. 双循环算出拟牛顿方向 d
        Eigen::VectorXd d = compute_lbfgs_direction(g, hist);

        // 2. 强 Wolfe 线搜索确定黄金步长
        double alpha = line_search.search(x, d, 1.0);

        std::cout << std::setw(14) << alpha << " | (" << x[0] << ", " << x[1] << ")\n";

        // 3. 推进状态并记录曲率
        Eigen::VectorXd x_next = x + alpha * d;
        Eigen::VectorXd g_next = grad_func(x_next);

        hist.push(x_next - x, g_next - g);
        x = x_next;
    }

    std::cout << "\n>> 理论最优解: (1.000000, 1.000000), 理论最低 Cost: 0.000000\n";
    std::cout << ">> 实际计算解: (" << x[0] << ", " << x[1] << "), 实际最终 Cost: " << cost_func(x) << "\n";

    return 0;
}