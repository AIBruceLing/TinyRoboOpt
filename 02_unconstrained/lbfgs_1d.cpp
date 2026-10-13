#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <iomanip>

// =========================================================================
// 1. 测试一维非线性目标函数: f(x) = (x - 2)^4 + 2 * (x - 2)^2
//    极小值点显然在 x* = 2.0 处，此时 f(2.0) = 0
// =========================================================================
double cost_func(double x) {
    double t = x - 2.0;
    return std::pow(t, 4) + 2.0 * t * t;
}

// 一阶导数（梯度）: f'(x) = 4 * (x - 2)^3 + 4 * (x - 2)
double grad_func(double x) {
    double t = x - 2.0;
    return 4.0 * std::pow(t, 3) + 4.0 * t;
}

// =========================================================================
// 2. 一维 L-BFGS 历史曲率记忆队列 (全部是纯 double 浮点数)
// =========================================================================
struct LBFGSHistory1D {
    int m = 5; // 记忆最近 5 步
    std::deque<double> s_list;   // 位移差 s_k = x_{k+1} - x_k
    std::deque<double> y_list;   // 导数差 y_k = g_{k+1} - g_k
    std::deque<double> rho_list; // rho_k = 1 / (s_k * y_k)

    void push(double s, double y) {
        double sy = s * y;
        // 正定性保护: s 和 y 必须同号 (s * y > 0)，保证下坡曲率有效
        if (sy > 1e-10) {
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
// 3. 核心算法: 一维 L-BFGS 双循环递归 (Two-Loop Recursion)
// =========================================================================
double compute_lbfgs_direction_1d(double g, const LBFGSHistory1D& hist) {
    int k = hist.s_list.size();
    if (k == 0) {
        return -g; // 第一步没有历史，直接沿着负梯度方向出发
    }

    double q = g;
    std::vector<double> alpha(k);

    // 【第一层循环】：从最新的历史向最旧的历史倒推
    for (int i = k - 1; i >= 0; --i) {
        alpha[i] = hist.rho_list[i] * (hist.s_list[i] * q);
        q -= alpha[i] * hist.y_list[i];
    }

    // 【初始海森逆缩放】: gamma = (s * y) / (y * y) = s / y
    double s_last = hist.s_list.back();
    double y_last = hist.y_list.back();
    double gamma = (s_last * y_last) / (y_last * y_last); // 恰好就是最近一步的割线斜率倒数！
    
    double r = gamma * q;

    // 【第二层循环】：从最旧的历史向最新的历史正推
    for (int i = 0; i < k; ++i) {
        double beta = hist.rho_list[i] * (hist.y_list[i] * r);
        r += hist.s_list[i] * (alpha[i] - beta);
    }

    return -r; // 返回经过拟牛顿二阶曲率修正后的搜索方向 d
}

// =========================================================================
// 4. 一维回溯线搜索 (满足 Armijo 充分下降条件)
// =========================================================================
double line_search_armijo_1d(double x, double d, double g, double alpha_init = 1.0) {
    double alpha = alpha_init;
    double c1 = 1e-4;
    double current_cost = cost_func(x);
    double initial_slope = g * d; // 初始一维斜率 phi'(0) = g * d

    if (initial_slope >= 0.0) {
        std::cerr << "Warning: direction is not descent!\n";
        return 0.0;
    }

    // Armijo 准则: f(x + alpha * d) <= f(x) + c1 * alpha * (g * d)
    while (cost_func(x + alpha * d) > current_cost + c1 * alpha * initial_slope) {
        alpha *= 0.5; // 步长折半回溯
        if (alpha < 1e-10) break;
    }
    return alpha;
}

// =========================================================================
// 5. 主循环测试
// =========================================================================
int main() {
    std::cout << "=================================================================\n";
    std::cout << "          TinyRoboOpt: Pure 1D L-BFGS Optimization Demo          \n";
    std::cout << "=================================================================\n";

    // 初始起点远离极小值点: x0 = 8.0 (极小值在 2.0)
    double x = 8.0;
    LBFGSHistory1D hist;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "起点 x0 = " << x << ", 初始 Cost = " << cost_func(x) << "\n\n";
    std::cout << "Iter |      Cost      |      Grad g     |   Step alpha   |   Current x\n";
    std::cout << "-----------------------------------------------------------------\n";

    for (int iter = 0; iter < 20; ++iter) {
        double g = grad_func(x);
        double cost = cost_func(x);

        std::cout << std::setw(4) << iter << " | "
                  << std::setw(14) << cost << " | "
                  << std::setw(15) << g << " | ";

        // 终止判据: 导数绝对值足够接近 0
        if (std::abs(g) < 1e-5) {
            std::cout << "\n>>> 优化收敛！梯度接近 0，成功找到极小值点！\n";
            break;
        }

        // 1. 递归计算 L-BFGS 下降方向 d
        double d = compute_lbfgs_direction_1d(g, hist);

        // 2. 线搜索确定步长 alpha
        double alpha = line_search_armijo_1d(x, d, g, 1.0);

        std::cout << std::setw(14) << alpha << " | " 
                  << std::setw(11) << x << "\n";

        // 3. 状态更新并记录差值
        double x_next = x + alpha * d;
        double g_next = grad_func(x_next);

        double s = x_next - x; // 位移差 Δx
        double y = g_next - g; // 导数差 Δg

        // 4. 将历史曲率压入队列
        hist.push(s, y);

        x = x_next;
    }

    std::cout << "\n>> 理论最优解: 2.000000, 理论最低 Cost: 0.000000\n";
    std::cout << ">> 实际计算解: " << x << ", 实际最终 Cost: " << cost_func(x) << "\n";

    return 0;
}