#include <iostream>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>

int main() {
    std::cout << "==========================================\n";
    std::cout << "       TinyRoboOpt: Kaczmarz Algorithm    \n";
    std::cout << "==========================================\n";

    // 1. 定义方程组: A * x = b (2 个方程, 3 个未知数, 矮胖矩阵)
    Eigen::MatrixXd A(2, 3);
    Eigen::VectorXd b(2);

    A << 1.0,  1.0, 1.0,   // 方程 1:  x1 + x2 +  x3 = 6
         1.0, -1.0, 2.0;   // 方程 2:  x1 - x2 + 2x3 = 4

    b << 6.0, 
         4.0;

    int m = A.rows(); // 2 个方程 (2 个超平面)
    int n = A.cols(); // 3 个未知数

    // 2. 初始点设为原点 (0, 0, 0)
    // 从原点出发是求出【最小范数解】的必要条件！
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    std::cout << "起点 x0: [" << x.transpose() << "]\n\n";

    // 3. Kaczmarz 算法核心迭代循环
    int max_iter = 30; // 迭代次数
    double tol = 1e-6; // 误差容忍度

    std::cout << std::fixed << std::setprecision(5);

    for (int iter = 0; iter < max_iter; ++iter) {
        // 轮流选择某一个方程 (第 i 行) 做正交投影
        int i = iter % m;

        Eigen::VectorXd a_i = A.row(i); // 第 i 个超平面的法向量
        double b_i = b[i];              // 第 i 个方程的目标值

        // 计算当前点到第 i 个方程的标量残差: error = b_i - a_i * x
        double error = b_i - a_i.dot(x);

        // ★★★ Kaczmarz 投影更新公式 ★★★
        // 沿着垂直法向 a_i 跨出一步，直接落在该超平面上！
        x = x + (error / a_i.squaredNorm()) * a_i;

        // 打印每一步的坐标与当前的全局残差模长
        double total_residual = (A * x - b).norm();
        std::cout << "Iter " << std::setw(2) << iter + 1 
                  << " (投影到方程 " << i + 1 << ") | "
                  << "当前解: [" << x[0] << ", " << x[1] << ", " << x[2] << "] | "
                  << "全局残差: " << total_residual << "\n";

        // 如果两条线索都满足了，提前收敛退出
        if (total_residual < tol) {
            std::cout << "\n>>> 在第 " << iter + 1 << " 步成功收敛！<<<\n";
            break;
        }
    }

    // 4. 验证对比：用解析公式 x* = A^T * (A * A^T)^(-1) * b 计算理论最小范数解
    Eigen::MatrixXd AAt = A * A.transpose();
    Eigen::VectorXd x_theoretical = A.transpose() * AAt.inverse() * b;

    std::cout << "\n------------------------------------------\n";
    std::cout << "Kaczmarz 迭代解 : [" << x.transpose() << "]\n";
    std::cout << "理论最小范数解   : [" << x_theoretical.transpose() << "]\n";
    std::cout << "两者误差模长     : " << (x - x_theoretical).norm() << "\n";
    std::cout << "------------------------------------------\n";

    return 0;
}