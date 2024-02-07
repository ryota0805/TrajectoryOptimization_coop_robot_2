from param import Parameter as p
import GenerateInitialPath
import util
import constraints
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import objective_function 
import plot
import time
import animation
import csv
import os

####手動でヤコビアンを計算しSQPを実行するプログラム
p.div=[p.N*i for i in range(p.M+1)]
# 計測開始
start_time = time.time()
#WayPointから設計変数の初期値を計算する
#cubicX, cubicY = GenerateInitialPath.cubic_spline()
cubicX, cubicY = GenerateInitialPath.cubic_spline_by_waypoint(p.WayPoint)
x, y, theta, theta1, theta2,omega1, omega2, v1, v2 = GenerateInitialPath.generate_initialpath2(cubicX, cubicY)
#x, y, theta, phi, v = GenerateInitialPath.generate_initialpath_randomly(cubicX, cubicY)
xs, ys, thetas, thetas1, thetas2, omega1, omega2, v1, v2 = GenerateInitialPath.initial_zero(0.1)
dt = np.array([0.5])
trajectory_matrix = np.array([x, y, theta, theta1, theta2, omega1, omega2, v1, v2])
trajectory_vector = util.matrix_to_vector(trajectory_matrix)
trajectory_vector = np.hstack([trajectory_vector, dt])

_,_,_,_,_,_,_,_,_,dt = util.generate_result(trajectory_vector)
print(dt)
#目的関数の設定
func = objective_function.objective_function_ex2
jac_of_objective_function = objective_function.jac_of_objective_function_ex2

#args = (5, 1, 20, 1)
args = (1, 5, 10, 1)

#制約条件の設定
cons = constraints.generate_cons_with_jac()

#変数の範囲の設定
bounds = constraints.generate_bounds()

#オプションの設定
options = {'maxiter':1000, 'ftol': 1e-6}


#最適化を実行
result = optimize.minimize(func, trajectory_vector, args = args, method='SLSQP', jac = jac_of_objective_function, constraints=cons, bounds=bounds, options=options)

# 計測終了
end_time = time.time()

#最適化結果の表示
print(result)
x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(result.x)
x1 = x - p.d1*np.cos(theta1) -p.d2/2*np.cos(theta)
y1 = y - p.d1*np.sin(theta1) -p.d2/2*np.sin(theta)
x2 = x + p.d1*np.cos(theta2) +p.d2/2*np.cos(theta)
y2 = y + p.d1*np.sin(theta2) +p.d2/2*np.sin(theta)

phi1 = theta1 - theta
phi2 = -(theta2 - theta)

file_path = 'thesis_data1/experiment5/ex5.7'
#file_path = None
time_list = [dt*i for i in range(p.N)]
print(time_list)
plot.vis_env()
plot.vis_path(x, y, file_path=file_path)
plot.compare_path(x1, y1, x2, y2, file_path=file_path)
"""
plot.vis_history_theta(theta, range_flag = True, file_path=file_path)
plot.history_robot_theta(theta1, theta2, range_flag = True, file_path=file_path)
plot.history_robot_phi(phi1, phi2, range_flag = True, file_path=file_path)
plot.history_robot_omega(omega1, omega2, range_flag = True, file_path=file_path)
plot.history_robot_v(v1, v2, range_flag = True, file_path=file_path)
"""
path_length = 0
for i in range(p.N-1):
    path_length += ((x[i] - x[i+1])**2 + (y[i] - y[i+1])**2)**(0.5)

print("path_length:{}".format(path_length))

# 経過時間を計算
elapsed_time = end_time - start_time
print(f"実行時間: {elapsed_time}秒")
print("dt={}, tf={}".format(dt, time_list[-1]))

#評価関数の計算
J1, J2, J3, J4 = objective_function.evaluation_ex2(result.x, *args)

animation.gen_robot_movie(x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, is_interpolation=True, vis_v=True, vis_robot_path=True)
animation.gen_robot_fig(x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, is_interpolation=True, vis_v=True, vis_robot_path=True)
data_list = [x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, time_list]

#データの保存
for data in data_list:
    with open(file_path+'/data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
with open(file_path+'/data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            t_data = ['dt', 't_f', dt, time_list[-1]]
            path_data = ['length', path_length]
            J_list = ['J_opt', 'J1', 'J2', 'J3', 'J4', result.fun, J1, J2, J3, J4]
            iter_list = ['iter', result.nit]
            w_list = ['w1', 'w2', 'w3', 'w4', args[0], args[1], args[2], args[3]]
            writer.writerow(t_data)
            writer.writerow(path_data)
            writer.writerow(J_list)
            writer.writerow(iter_list)
            writer.writerow(w_list)
