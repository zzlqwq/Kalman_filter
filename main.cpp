#include <iostream>
#include "Eigen\Dense"
#include "kalman.h"
#include <fstream>

using namespace std;
#define N 1000
#define T 0.01

double data_x[N], data_y[N];


float sample(float x0, float v0, float acc, float t) {
  return x0 + v0 * t + 0.5 * acc * t * t;
}

float GetRand() {
  return 0.5 * rand() / RAND_MAX - 0.25;
}

int main() {
  ofstream fout1, fout2;
  fout1.open("../truedata.txt");
  fout2.open("../data.txt");
  float t;
  float vx = -4, vy = 6.5;
  float ax = 0, ay = 0;
  for (int i = 0; i < N; i++) {
	t = i * T;
	float true_x = sample(0, vx, ax, t);
	float true_y = sample(0, vy, ay, t);
	data_x[i] =  true_x + GetRand();
	data_y[i] =  true_y + GetRand();
	fout1 << true_x << " " << true_y << " " << vx << " " << vy << " "<< ax << " "<< ay << std::endl;
  }
  fout1.close();
  int stateSize = 6;
  int measSize = 2;
  int inputSize = 0;
  KalmanFilter kf(stateSize, measSize, inputSize);
  /// 初始化A & B矩阵
  Eigen::MatrixXd A(stateSize, stateSize);
  A << 1, 0, T, 0, 0.5 * T * T, 0,
	  0, 1, 0, T, 0, 0.5 * T * T,
	  0, 0, 1, 0, T, 0,
	  0, 0, 0, 1, 0, T,
	  0, 0, 0, 0, 1, 0,
	  0, 0, 0, 0, 0, 1;
  Eigen::MatrixXd B(0, 0);
  /// 初始化H矩阵
  Eigen::MatrixXd H(measSize, stateSize);
  H << 1, 0, 0, 0, 0, 0,
	  0, 1, 0, 0, 0, 0;
  /// 初始化P矩阵
  Eigen::MatrixXd P(stateSize, stateSize);
  P.setIdentity();
  /// 初始化Q & R矩阵
  Eigen::MatrixXd R(measSize, measSize);
  R.setIdentity() * 0.01;
  Eigen::MatrixXd Q(stateSize, stateSize);
  Q.setIdentity() * 0.01;

  Eigen::VectorXd x(stateSize);
  x << 0, 0, 0, 0, 0, 0;
  Eigen::VectorXd u(0);
  Eigen::VectorXd z(measSize);
  z.setZero();

  Eigen::VectorXd res(stateSize);
  kf.init(x, P, R, Q, A, B, H);

  for (int i = 0; i < N; i++) {
	kf.predict();
	z << data_x[i], data_y[i];
	res << kf.update(z);
	fout2 << res[0] << " " << res[1] << " " << res[2] << " " << res[3] << " "
		  << res[4] << " " << res[5] << endl;
  }
  fout2.close();
  return 0;
}