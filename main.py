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

####シミュレーション用の経路計画

def main(start, goal, start_theta, goal_theta, N=p.N):
    p.initial_x, p.initial_y = start[0], start[1]
    p.terminal_x, p.terminal_y = goal[0], goal[1]
    p.initial_theta, p.terminal_theta = start_theta, goal_theta
    p.N = N
    
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
    func = objective_function.objective_function4
    jac_of_objective_function = objective_function.jac_of_objective_function4

    args = (1, 1)

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

    time_list = [dt*i for i in range(p.N)]
    plot.vis_env()
    plot.vis_path(x, y)
    plot.compare_path(x1, y1, x2, y2)
    plot.vis_history_theta(theta, range_flag = True)
    plot.history_robot_theta(theta1, theta2, range_flag = True)
    plot.history_robot_phi(phi1, phi2, range_flag = True)
    plot.history_robot_omega(omega1, omega2, range_flag = True)
    plot.history_robot_v(v1, v2, range_flag = True)


    # 経過時間を計算
    elapsed_time = end_time - start_time
    print(f"実行時間: {elapsed_time}秒")
    print("dt={}, tf={}".format(dt, time_list[-1]))

    #animation.gen_robot_movie(x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, is_interpolation=True, vis_v=True, vis_robot_path=True)
    
    return x, y, theta
