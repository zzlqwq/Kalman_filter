//
// Created by 11706 on 2022/1/26.
//

#ifndef KALMAN_FILTER__KALMAN_H
#define KALMAN_FILTER__KALMAN_H

#include <Eigen\Dense>

class KalmanFilter {
 private:
  int stateSize; ///state variable's dimension
  int measSize; ///measurement variable's dimension
  int uSize; ///input variable's dimension
  Eigen::VectorXd x;
  Eigen::VectorXd z;
  Eigen::VectorXd u;

  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd P; ///covariance
  Eigen::MatrixXd H;
  Eigen::MatrixXd R; ///measurement noise covariance
  Eigen::MatrixXd Q; ///process noise covariance
 public:
  KalmanFilter(int stateSize_, int measSize_, int uSize_);
  ~KalmanFilter() = default;
  void init(Eigen::VectorXd &x_,
			Eigen::MatrixXd &P_,
			Eigen::MatrixXd &R_,
			Eigen::MatrixXd &Q_,
			Eigen::MatrixXd &A_,
			Eigen::MatrixXd &B_,
			Eigen::MatrixXd &H_);
  Eigen::VectorXd predict();
  Eigen::VectorXd predict(Eigen::VectorXd &u_);
  Eigen::VectorXd update(const Eigen::VectorXd &z_meas);
};
#endif //KALMAN_FILTER__KALMAN_H
