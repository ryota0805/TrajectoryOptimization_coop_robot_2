#パラメータ管理class
import numpy as np
import env

class Parameter:
    env_data = env.Env()
    N = 30                                                     #系列データの長さ
    M = 9                                                       #設計変数の種類の個数

    #変数を分離するためのリスト
    div = []
    for i in range(M+1):
        div.append(N*i)
    
    #初期状態と終端状態
    set_cons = {'initial_x'     :True,                          #境界条件をセットするかどうか
                'terminal_x'    :True, 
                'initial_y'     :True, 
                'terminal_y'    :True, 
                'initial_theta' :True, 
                'terminal_theta':True, 
                'initial_theta1' :True,
                'terminal_theta1':True,
                'initial_theta2' :True, 
                'terminal_theta2':True,
                'initial_omega1' :False,
                'terminal_omega1':True,
                'initial_omega2' :False,
                'terminal_omega2':True,
                'initial_v1'     :False, 
                'terminal_v1'    :True,
                'initial_v2'     :False, 
                'terminal_v2'    :True,
                }
    
    initial_x = 13.5                                             #x[m]
    terminal_x = 10                                         #x[m]
    
    initial_y = -8                                             #y[m]
    terminal_y = -1                                              #y[m]
    
    initial_theta = np.pi/2                                          #theta[rad]
    terminal_theta = 0                                          #theta[rad]

    initial_theta1 = initial_theta                                         #theta[rad]
    terminal_theta1 = terminal_theta                                          #theta[rad]
    
    initial_theta2 = initial_theta
    terminal_theta2 = terminal_theta
    
    initial_omega1 = 0
    terminal_omega1 = 0
    
    initial_omega2 = 0
    terminal_omega2 = 0
    
    initial_v1 = 0                                               #v[m/s]
    terminal_v1 = 0                                              #v[m/s]
    
    initial_v2 = 0                                               #v[m/s]
    terminal_v2 = 0                                              #v[m/s]
    
    #変数の範囲
    x_min = env_data.x_range[0]                                                  #x[m]
    x_max = env_data.x_range[1]                                                  #x[m]
    y_min = env_data.y_range[0]                                                 #y[m]
    y_max = env_data.y_range[1]                                                  #y[m]
    theta_min = -np.pi * 180/ 180                                         #theta[rad]
    theta_max = np.pi *180/180    
    theta1_min = -np.pi * 180/ 180                                         #theta[rad]
    theta1_max = np.pi *180/180                                          #tehta[rad]
    theta2_min = -np.pi * 180/ 180                                           #theta[rad]
    theta2_max = np.pi * 180/ 180  
    omega1_min = -np.pi/8
    omega1_max = np.pi/8
    omega2_min = -np.pi/8
    omega2_max = np.pi/8
    v1_min = -0.5                                                   #v[m/s]
    v1_max = 0.5                                                   #v[m/s]
    v2_min = -0.5                                                   #v[m/s]
    v2_max = 0.5                                                   #v[m/s]
    dt_min = 0
    dt_max = 4
    
    #ロボットの把持部分の最大値
    phi_max = np.pi/6
    WayPoint = np.array([[initial_x, initial_y],                #初期パス　[x, y] 
                        [15, 0],
                        [13, 0],
                        [terminal_x, terminal_y]])     



    dt = 1                                                      #刻み幅[s]                                             
    d1 = 0
    d2 = 0.6+0.25*2

    #障害物のパラメータ 
    #　(x, y, r)
    #　x　: 円の中心座標
    #　y　: 円の中心座標
    #　r　: 半径 
    obstacle_list = [(10, -1, 3), (20, 1, 3)]                   #障害物のパラメータが格納されたリスト
    
    
    #wallのパラメータ
    wall_thick = 1                                #wallの厚さ
    margin = 2
    
    #robot size
    robot_size = 1
    
    #v, omegaの初期値との誤差
    error_omega = 0.001
    error_v = 0.001
