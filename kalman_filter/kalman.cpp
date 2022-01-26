//
// Created by 11706 on 2022/1/26.
//
#include "kalman.h"
#include "iostream"

KalmanFilter::KalmanFilter(int stateSize_, int measSize_, int uSize_)
	: stateSize(stateSize_), measSize(measSize_), uSize(uSize_) {
  if (stateSize == 0 || measSize == 0) {
	std::cerr << "Error, State size and measurement size must bigger than 0\n";
  }

  x.resize(stateSize);
  x.setZero();

  A.resize(stateSize, stateSize);
  A.setIdentity();

  u.resize(uSize);
  u.transpose();
  u.setZero();

  B.resize(stateSize, uSize);
  B.setZero();

  P.resize(stateSize, stateSize);
  P.setIdentity();

  H.resize(measSize, stateSize);
  H.setZero();

  z.resize(measSize);
  z.setZero();

  Q.resize(stateSize, stateSize);
  Q.setZero();

  R.resize(measSize, measSize);
  R.setZero();
}

void KalmanFilter::init(Eigen::VectorXd &x_,
						Eigen::MatrixXd &P_,
						Eigen::MatrixXd &R_,
						Eigen::MatrixXd &Q_,
						Eigen::MatrixXd &A_,
						Eigen::MatrixXd &B_,
						Eigen::MatrixXd &H_) {
  x = x_;
  P = P_;
  R = R_;
  Q = Q_;
  A = A_;
  B = B_;
  H = H_;
}

Eigen::VectorXd KalmanFilter::predict(Eigen::VectorXd &u_) {
  u = u_;
  x = A * x + B * u;
  Eigen::MatrixXd A_T = A.transpose();
  P = A * P * A_T + Q;
  return x;
}

Eigen::VectorXd KalmanFilter::predict() {
  x = A * x;
  Eigen::MatrixXd A_T = A.transpose();
  P = A * P * A_T + Q;
  //std::cout << "P = " << P << std::endl;
  return x;
}

Eigen::VectorXd KalmanFilter::update(const Eigen::VectorXd &z_meas){
  /// step 1 : 计算卡尔曼增益
  /// K_Molecular K_Denominator 分别表示卡尔曼增益计算公式的分子与分母部分
  Eigen::MatrixXd K_Molecular, K_Denominator, K;
  Eigen::MatrixXd H_T = H.transpose();
  K_Molecular = P * H_T;
  K_Denominator = H * P * H_T + R;
  Eigen::MatrixXd K_Denominator_inv = K_Denominator.inverse();
  K = K_Molecular * K_Denominator_inv;

  /// step 2 : 更新最优值x
  x = x + K * (z_meas - H * x);
  /// step 3 : 更新协方差矩阵P
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(stateSize, stateSize);
  P = (I - K * H) * P;
  return x;
}
