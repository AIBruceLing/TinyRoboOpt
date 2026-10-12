#include <iostream>
#include <Eigen/Dense>

int main() {
    std::cout << "==========================================\n";
    std::cout << "       TinyRoboOpt: Least Squares (LS)    \n";
    std::cout << "==========================================\n";

    // 假设我们采样到了 5 组二维实验数据点 (x, y)
    // 理想方程大约是 y = 2.0 * x + 1.0，但带有观测噪声
    // 点：(1, 3.1), (2, 4.9), (3, 7.2), (4, 8.8), (5, 11.1)
    int m = 5; // 数据点个数 (m >= n)
    int n = 2; // 未知数个数 (斜率 k 和截距 c)

    Eigen::MatrixXd A(m, n);
    Eigen::VectorXd b(m);

    A << 1.0, 1.0,
         2.0, 1.0,
         3.0, 1.0,
         4.0, 1.0,
         5.0, 1.0;

    b << 3.1,
         4.9,
         7.2,
         8.8,
         11.1;

    std::cout << "系数矩阵 A (维度 " << A.rows() << "x" << A.cols() << "):\n" << A << "\n\n";
    std::cout << "观测向量 b:\n" << b << "\n\n";

    // -------------------------------------------------------------
    // 方法 1：基于正规方程的解析推导 (A^T * A)^(-1) * A^T * b
    // -------------------------------------------------------------
    Eigen::MatrixXd AtA = A.transpose() * A;
    Eigen::VectorXd Atb = A.transpose() * b;

    // 注意：工程上切忌直接调用 .inverse()，推荐用 ldlt() 求解对称正定系统，兼顾精度与速度
    Eigen::VectorXd x_star_normal = AtA.ldlt().solve(Atb);

    std::cout << ">>> [方法 1: 正规方程解法 (A^T*A)^(-1) * A^T*b] <<<\n";
    std::cout << "斜率 k: " << x_star_normal[0] << "\n";
    std::cout << "截距 c: " << x_star_normal[1] << "\n\n";

    // -------------------------------------------------------------
    // 方法 2：工业标准写法 (QR 分解直接求解 Ax = b)
    // 数值稳定性优于方法 1 (避免了 A^T*A 造成条件数平方放大)
    // -------------------------------------------------------------
    Eigen::VectorXd x_star_qr = A.colPivHouseholderQr().solve(b);

    std::cout << ">>> [方法 2: 工业级 QR 分解解法] <<<\n";
    std::cout << "斜率 k: " << x_star_qr[0] << "\n";
    std::cout << "截距 c: " << x_star_qr[1] << "\n\n";

    // -------------------------------------------------------------
    // 验证残差: ||Ax - b||^2
    // -------------------------------------------------------------
    double residual_norm = (A * x_star_qr - b).squaredNorm();
    std::cout << "最终拟合的残差平方和 (Cost): " << residual_norm << "\n";

    return 0;
}