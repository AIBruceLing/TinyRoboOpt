#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <Eigen/Dense>

// ============================================================
// 1. 场景定义
// ============================================================
const Eigen::Vector2d OBSTACLE(2.0, 0.0); // 障碍物位置
const double R_SAFE = 0.8;                // 安全膨胀距离
const double V_MAX  = 1.5;                // 最大线速度 (m/s)

struct Pose {
    double x, y, theta;
};

// ============================================================
// 2. 解包函数
// ============================================================
void unpack(const Eigen::VectorXd& X, std::vector<Pose>& P, std::vector<double>& dt) {
    P.resize(5);
    dt.resize(4);

    P[0] = {0.0, 0.0, 0.0}; // 起点固定
    P[4] = {4.0, 0.0, 0.0}; // 终点固定

    for (int i = 0; i < 3; ++i) {
        P[i + 1] = {X[i * 3 + 0], X[i * 3 + 1], X[i * 3 + 2]};
    }
    for (int i = 0; i < 4; ++i) {
        dt[i] = X[9 + i];
    }
}

// ============================================================
// 3. 计算所有弹性残差
// ============================================================
Eigen::VectorXd compute_residuals(const Eigen::VectorXd& X) {
    std::vector<Pose> P;
    std::vector<double> dt;
    unpack(X, P, dt);

    std::vector<double> r;

    // 1. 避障弹簧
    double w_obs = 15.0;
    for (int i = 1; i <= 3; ++i) {
        double dist = std::hypot(P[i].x - OBSTACLE.x(), P[i].y - OBSTACLE.y());
        if (dist < R_SAFE) {
            r.push_back(w_obs * (R_SAFE - dist));
        } else {
            r.push_back(0.0);
        }
    }

    // 2. 差速底盘运动学非完整性约束
    double w_kin = 8.0;
    for (int i = 0; i < 4; ++i) {
        double dx = P[i + 1].x - P[i].x;
        double dy = P[i + 1].y - P[i].y;
        double th = P[i].theta;
        r.push_back(w_kin * (dx * std::sin(th) - dy * std::cos(th)));
    }

    // 3. 时间最优拉力
    double w_time = 1.0;
    for (int i = 0; i < 4; ++i) {
        r.push_back(w_time * dt[i]);
    }

    // 4. 速度上限
    double w_vel = 10.0;
    for (int i = 0; i < 4; ++i) {
        double dist = std::hypot(P[i + 1].x - P[i].x, P[i + 1].y - P[i].y);
        double v = dist / std::max(dt[i], 1e-4);
        if (v > V_MAX) {
            r.push_back(w_vel * (v - V_MAX));
        } else {
            r.push_back(0.0);
        }
    }

    // 5. 间距均匀性
    double w_dist = 2.0;
    for (int i = 0; i < 4; ++i) {
        double dist = std::hypot(P[i + 1].x - P[i].x, P[i + 1].y - P[i].y);
        r.push_back(w_dist * (dist - 1.0));
    }

    Eigen::VectorXd res(r.size());
    for (size_t i = 0; i < r.size(); ++i) res[i] = r[i];
    return res;
}

// ============================================================
// 4. 数值雅可比
// ============================================================
Eigen::MatrixXd compute_jacobian(const Eigen::VectorXd& X, const Eigen::VectorXd& r0) {
    int m = r0.size();
    int n = X.size();
    Eigen::MatrixXd J(m, n);
    double eps = 1e-5;

    for (int j = 0; j < n; ++j) {
        Eigen::VectorXd X_plus = X;
        X_plus[j] += eps;
        Eigen::VectorXd r_plus = compute_residuals(X_plus);
        J.col(j) = (r_plus - r0) / eps;
    }
    return J;
}

// ============================================================
// 5. 可视化辅助函数：导出 CSV 并直接拉起 Python 绘图
// ============================================================
void visualize(const std::vector<Pose>& init_poses, const std::vector<Pose>& opt_poses) {
    // 1. 写出轨迹数据到 CSV
    std::ofstream file("teb_trajectory.csv");
    file << "type,x,y,theta\n";
    for (const auto& p : init_poses) file << "init," << p.x << "," << p.y << "," << p.theta << "\n";
    for (const auto& p : opt_poses)  file << "opt,"  << p.x << "," << p.y << "," << p.theta << "\n";
    file.close();

    // 2. 自动生成一个一键可视化的 Python 脚本
    std::ofstream py_file("plot_teb.py");
    py_file << R"(
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('teb_trajectory.csv')
init_p = df[df['type'] == 'init']
opt_p = df[df['type'] == 'opt']

fig, ax = plt.subplots(figsize=(10, 5))

# 1. 绘制障碍物及其安全圈
obs_x, obs_y, r_safe = 2.0, 0.0, 0.8
obstacle_circle = plt.Circle((obs_x, obs_y), 0.2, color='crimson', label='Obstacle', zorder=5)
safe_zone = plt.Circle((obs_x, obs_y), r_safe, color='coral', alpha=0.25, linestyle='--', linewidth=1.5, label='Safety Zone')
ax.add_patch(obstacle_circle)
ax.add_patch(safe_zone)

# 2. 绘制初始直线路网 (撞障轨迹)
ax.plot(init_p['x'], init_p['y'], 'k--', alpha=0.5, label='Initial Trajectory (Collision)')
ax.scatter(init_p['x'], init_p['y'], color='gray', s=40)

# 3. 绘制 TEB 优化后的平滑避障轨迹
ax.plot(opt_p['x'], opt_p['y'], '-o', color='royalblue', linewidth=2.5, label='Optimized TEB Trajectory')

# 4. 用箭头画出每个位姿点的车身朝向角 (theta)
scale = 0.3
for _, row in opt_p.iterrows():
    dx = scale * np.cos(row['theta'])
    dy = scale * np.sin(row['theta'])
    ax.arrow(row['x'], row['y'], dx, dy, head_width=0.08, head_length=0.1, fc='blue', ec='blue', zorder=4)

# 5. 美化图表
ax.set_title("Tiny-TEB: Spatio-temporal Obstacle Avoidance Optimization", fontsize=13, fontweight='bold')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-1.2, 1.5)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
)";
    py_file.close();

    std::cout << "\n>>> 正在自动拉起 Python 可视化窗口... <<<\n";
    // 如果你在有 GUI 的桌面环境（如 Ubuntu 桌面），直接唤起绘图窗口
    int ret = std::system("python3 plot_teb.py");
    (void)ret;
}

// ============================================================
// 6. 主循环
// ============================================================
int main() {
    std::cout << "==========================================\n";
    std::cout << "     Tiny-TEB: Trajectory with Visualizer \n";
    std::cout << "==========================================\n";

    // 初始轨迹参数 (穿过障碍物)
    Eigen::VectorXd X(13);
    X[0] = 1.0; X[1] = 0.01; X[2] = 0.0;
    X[3] = 2.0; X[4] = 0.01; X[5] = 0.0; // 穿过障碍 (2,0)
    X[6] = 3.0; X[7] = 0.01; X[8] = 0.0;
    X[9] = 1.0; X[10] = 1.0; X[11] = 1.0; X[12] = 1.0;

    // 记录初始位姿用于对比
    std::vector<Pose> init_poses;
    std::vector<double> init_dt;
    unpack(X, init_poses, init_dt);

    double mu = 1.0;

    // LM 优化循环
    for (int iter = 0; iter < 25; ++iter) {
        Eigen::VectorXd r = compute_residuals(X);
        double cost = 0.5 * r.squaredNorm();

        Eigen::MatrixXd J = compute_jacobian(X, r);
        Eigen::VectorXd grad = J.transpose() * r;
        Eigen::MatrixXd H = J.transpose() * J;

        // 解 (J^T*J + mu*I) * d = -grad
        Eigen::VectorXd d = (H + mu * Eigen::MatrixXd::Identity(13, 13)).ldlt().solve(-grad);

        Eigen::VectorXd X_new = X + d;
        for (int i = 9; i < 13; ++i) {
            if (X_new[i] < 0.05) X_new[i] = 0.05;
        }

        Eigen::VectorXd r_new = compute_residuals(X_new);
        double cost_new = 0.5 * r_new.squaredNorm();

        if (cost_new < cost) {
            X = X_new;
            mu *= 0.7;
        } else {
            mu *= 2.0;
        }
    }

    std::vector<Pose> opt_poses;
    std::vector<double> opt_dt;
    unpack(X, opt_poses, opt_dt);

    std::cout << ">>> 优化完成！即将打开图形展示！<<<\n";

    // 唤起可视化
    visualize(init_poses, opt_poses);

    return 0;
}