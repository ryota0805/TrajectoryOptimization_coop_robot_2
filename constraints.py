#不等式制約、等式制約を定義する
from param import Parameter as p
import util
import numpy as np
import env

########
#bounds(変数の範囲)を設定する関数
########

#変数の数だけタプルのリストとして返す関数
def generate_bounds():
    
    #boundsのリストを生成
    bounds = []
    
    #xの範囲
    for i in range(p.N):
        bounds.append((p.x_min, p.x_max))
        
    #yの範囲
    for i in range(p.N):
        bounds.append((p.y_min, p.y_max))
        
    #thetaの範囲
    for i in range(p.N):
        bounds.append((p.theta_min, p.theta_max))
        
    #theta1の範囲
    for i in range(p.N):
        bounds.append((p.theta1_min, p.theta1_max))
        
    #theta2の範囲
    for i in range(p.N):
        bounds.append((p.theta2_min, p.theta2_max))
        
    #omega1の範囲
    for i in range(p.N):
        bounds.append((p.omega1_min, p.omega1_max))
        
    #omega2の範囲
    for i in range(p.N):
        bounds.append((p.omega2_min, p.omega2_max))
        
    #v1の範囲
    for i in range(p.N):
        bounds.append((p.v1_min, p.v1_max))
        
    #v2の範囲
    for i in range(p.N):
        bounds.append((p.v2_min, p.v2_max))
        
    #dtの範囲
    bounds.append((p.dt_min, p.dt_max))
        
    return bounds


def jac_of_constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x1':
            #jac_x
            jac_x[i] = -1
            jac_x[i+1] = 1

            #jac_theta
            jac_theta[i] = p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(1/2)*(theta[i+1] - theta[i]) + p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)
            jac_theta[i+1] = p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(1/2)*(theta[i+1] - theta[i]) + p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)
            
            #jac_theta1
            jac_theta1[i] = (v1[i]/2 + v1[i+1]/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)*(1/2)*dt
            jac_theta1[i+1] = (v1[i]/2 + v1[i+1]/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)*(1/2)*dt
            
            #jac_v1
            jac_v1[i] = -(1/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)*dt
            jac_v1[i+1] = -(1/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)*dt
            
            #jac_dt
            jac_dt = -(v1[i]/2 + v1[i+1]/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)
        
        elif args[1][0] == 'y1':
            #jac_y
            jac_y[i] = -1
            jac_y[i+1] = 1
            
            #jac_theta
            jac_theta[i] = -p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)*(1/2)*(theta[i+1] - theta[i]) - p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(-1)
            jac_theta[i+1] = -p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)*(1/2)*(theta[i+1] - theta[i]) - p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)
            
            #jac_theta1
            jac_theta1[i] = -(v1[i]/2 + v1[i+1]/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)*(1/2)*dt
            jac_theta1[i+1] = -(v1[i]/2 + v1[i+1]/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)*(1/2)*dt
            
            #jac_v1
            jac_v1[i] = -(1/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)*dt
            jac_v1[i+1] = -(1/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)*dt
            
            #jac_dt
            jac_dt = -(v1[i]/2 + v1[i+1]/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)
            
        elif args[1][0] == 'theta1':
            #jac_theta1
            jac_theta1[i] = -1
            jac_theta1[i+1] = 1

            #jac_omega1
            jac_omega1[i] = -(1/2)*dt
            jac_omega1[i+1] = -(1/2)*dt
            
            #jac_dt
            jac_dt = -(omega1[i]/2 + omega1[i+1]/2)
            
            
        elif args[1][0] == 'x2':
            #jac_x
            jac_x[i] = -1
            jac_x[i+1] = 1

            #jac_theta
            jac_theta[i] = -p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(1/2)*(theta[i+1] - theta[i]) - p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)
            jac_theta[i+1] = -p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(1/2)*(theta[i+1] - theta[i]) - p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)
            
            #jac_theta2
            jac_theta2[i] = (v2[i]/2 + v2[i+1]/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)*(1/2)*dt
            jac_theta2[i+1] = (v2[i]/2 + v2[i+1]/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)*(1/2)*dt
            
            #jac_v2
            jac_v2[i] = -(1/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)*dt
            jac_v2[i+1] = -(1/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)*dt
            
            #jac_dt
            jac_dt = -(v2[i]/2 + v2[i+1]/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)
        
        elif args[1][0] == 'y2':
            #jac_y
            jac_y[i] = -1
            jac_y[i+1] = 1
            
            #jac_theta
            jac_theta[i] = p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)*(1/2)*(theta[i+1] - theta[i]) + p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(-1)
            jac_theta[i+1] = p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(-1)*(1/2)*(theta[i+1] - theta[i]) + p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)
            
            #jac_theta2
            jac_theta2[i] = -(v2[i]/2 + v2[i+1]/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)*(1/2)*dt
            jac_theta2[i+1] = -(v2[i]/2 + v2[i+1]/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)*(1/2)*dt
            
            #jac_v2
            jac_v2[i] = -(1/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)*dt
            jac_v2[i+1] = -(1/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)*dt
            
            #jac_dt
            jac_dt = -(v2[i]/2 + v2[i+1]/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)
            
        elif args[1][0] == 'theta2':
            #jac_theta2
            jac_theta2[i] = -1
            jac_theta2[i+1] = 1

            #jac_omega2
            jac_omega2[i] = -(1/2)*dt
            jac_omega2[i+1] = -(1/2)*dt
            
            #jac_dt
            jac_dt = -(omega2[i]/2 + omega2[i+1]/2)
            
        else:
            return 'Error'
        
        #ベクトルに直す
        jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
        return jac_cons
    
    elif args[0] == 'avoid_obstacle':
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            jac_x[i] = 10 * ((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 9
            jac_y[i] = 10 * ((2*0.8/obs_rectangle[k][3]) ** 10) * (y[i] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 9
            
            ##ベクトルに直す
            jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
        
            return jac_cons
            
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            jac_x[i] = 2 * (x[i] - obs_circle[k][0])
            jac_y[i] = 2 * (y[i] - obs_circle[k][1])
            
            #ベクトルに直す
            jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
        
            return jac_cons
    
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x':
            if ini_ter == 'ini':
                jac_x[0] = 1
            
            elif ini_ter == 'ter':
                jac_x[-1] = 1
                
        elif variable == 'y':
            if ini_ter == 'ini':
                jac_y[0] = 1
            
            elif ini_ter == 'ter':
                jac_y[-1] = 1  
                
        elif variable == 'theta':
            if ini_ter == 'ini':
                jac_theta[0] = 1
            
            elif ini_ter == 'ter':
                jac_theta[-1] = 1 
                
        elif variable == 'theta1':
            if ini_ter == 'ini':
                jac_theta1[0] = 1
            
            elif ini_ter == 'ter':
                jac_theta1[-1] = 1  
                
        elif variable == 'theta2':
            if ini_ter == 'ini':
                jac_theta2[0] = 1
            
            elif ini_ter == 'ter':
                jac_theta2[-1] = 1  
                
        elif variable == 'omega1':
            if ini_ter == 'ini':
                jac_omega1[0] = 1
            
            elif ini_ter == 'ter':
                jac_omega1[-1] = 1
                
        elif variable == 'omega2':
            if ini_ter == 'ini':
                jac_omega2[0] = 1
            
            elif ini_ter == 'ter':
                jac_omega2[-1] = 1
                
        elif variable == 'v1':
            if ini_ter == 'ini':
                jac_v1[0] = 1
            
            elif ini_ter == 'ter':
                jac_v1[-1] = 1    
                
        elif variable == 'v2':
            if ini_ter == 'ini':
                jac_v2[0] = 1
            
            elif ini_ter == 'ter':
                jac_v2[-1] = 1    
        
        #ベクトルに直す
        jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
        return jac_cons
    
    #把持部分の角度の制約
    elif args[0] == 'steer':
        variable, i = args[1][0], args[1][1]
        if variable == 'theta1':
            jac_theta[i] =  -2*(theta[i] - theta1[i])
            jac_theta1[i] = 2*(theta[i] - theta1[i])
            
        elif variable == 'theta2':
            jac_theta[i] =  -2*(theta[i] - theta2[i])
            jac_theta2[i] = 2*(theta[i] - theta2[i])
    
        #ベクトルに直す
        jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
        return jac_cons
    
    #不等式制約として与えるomega,vの初期値
    elif args[0] == 'ini':
        variable, i = args[1][0], args[1][1]
        if variable == 'omega1':
            jac_omega1[0] =  -2*(omega1[i] - p.initial_omega1)
            
        elif variable == 'omega2':
            jac_omega2[0] =  -2*(omega2[i] - p.initial_omega2)
            
        elif variable == 'v1':
            jac_v1[0] =  -2*(v1[i] - p.initial_v1)
            
        elif variable == 'v2':
            jac_v2[0] =  -2*(v2[i] - p.initial_v2)
    
        #ベクトルに直す
        jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
        return jac_cons
    
    
    
def constraint(x, *args):
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    if args[0] == 'model':
        i = args[1][1]
        if args[1][0] == 'x1':
            value = x[i+1] - x[i]  + p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(theta[i+1] - theta[i]) - (v1[i]/2 + v1[i+1]/2)*np.cos(theta1[i]/2 + theta1[i+1]/2)*dt
        
        elif args[1][0] == 'y1':
            value = y[i+1] - y[i]  - p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(theta[i+1] - theta[i]) - (v1[i]/2 + v1[i+1]/2)*np.sin(theta1[i]/2 + theta1[i+1]/2)*dt
            
        elif args[1][0] == 'theta1':
            value = theta1[i+1] - theta1[i] - (omega1[i]/2 + omega1[i+1]/2)*dt
            
        elif args[1][0] == 'x2':
            value = x[i+1] - x[i] - p.d2/2*np.sin(theta[i]/2 + theta[i+1]/2)*(theta[i+1] - theta[i]) - (v2[i]/2 + v2[i+1]/2)*np.cos(theta2[i]/2 + theta2[i+1]/2)*dt
        
        elif args[1][0] == 'y2':
            value = y[i+1] - y[i] + p.d2/2*np.cos(theta[i]/2 + theta[i+1]/2)*(theta[i+1] - theta[i]) - (v2[i]/2 + v2[i+1]/2)*np.sin(theta2[i]/2 + theta2[i+1]/2)*dt
            
        elif args[1][0] == 'theta2':
            value = theta2[i+1] - theta2[i] - (omega2[i]/2 + omega2[i+1]/2)*dt
             
        else:
            return 'Error'

        return value

    elif args[0] == 'avoid_obstacle':
        if args[1][0] == 'rectangle':
            k, i = args[1][1], args[1][2]
            
            value = (((2*0.8/obs_rectangle[k][2]) ** 10) * (x[i] - (obs_rectangle[k][0] + obs_rectangle[k][2]/2)) ** 10 + ((2*0.8/obs_rectangle[k][3]) ** 10) * (y[i] - (obs_rectangle[k][1] + obs_rectangle[k][3]/2)) ** 10) - 1
            
            return value
        
        elif args[1][0] == 'circle':
            k, i = args[1][1], args[1][2]
            
            value = ((x[i] - obs_circle[k][0]) ** 2 + (y[i] - obs_circle[k][1]) ** 2) - (obs_circle[k][2] + p.robot_size) ** 2

            return value 
    
    elif args[0] == 'boundary':
        variable, ini_ter = args[1][0], args[1][1]
        
        if variable == 'x':
            if ini_ter == 'ini':
                value = x[0] - p.initial_x
            
            elif ini_ter == 'ter':
                value = x[-1] - p.terminal_x
                
        elif variable == 'y':
            if ini_ter == 'ini':
                value = y[0] - p.initial_y
            
            elif ini_ter == 'ter':
                value = y[-1] - p.terminal_y  
                
        elif variable == 'theta':
            if ini_ter == 'ini':
                value = theta[0] - p.initial_theta
            
            elif ini_ter == 'ter':
                value = theta[-1] - p.terminal_theta
                
        elif variable == 'theta1':
            if ini_ter == 'ini':
                value = theta1[0] - p.initial_theta1
            
            elif ini_ter == 'ter':
                value = theta1[-1] - p.terminal_theta1
                
        elif variable == 'theta2':
            if ini_ter == 'ini':
                value = theta2[0] - p.initial_theta2
            
            elif ini_ter == 'ter':
                value = theta2[-1] - p.terminal_theta2
                
        elif variable == 'omega1':
            if ini_ter == 'ini':
                value = omega1[0] - p.initial_omega1
            
            elif ini_ter == 'ter':
                value = omega1[-1] - p.terminal_omega1
                
        elif variable == 'omega2':
            if ini_ter == 'ini':
                value = omega2[0] - p.initial_omega2
            
            elif ini_ter == 'ter':
                value = omega2[-1] - p.terminal_omega2
                
        elif variable == 'v1':
            if ini_ter == 'ini':
                value = v1[0] - p.initial_v1
            
            elif ini_ter == 'ter':
                value = v1[-1] - p.terminal_v1 
                
        elif variable == 'v2':
            if ini_ter == 'ini':
                value = v2[0] - p.initial_v2
            
            elif ini_ter == 'ter':
                value = v2[-1] - p.terminal_v2
    
        return value
    
    
    #把持部分の角度の制約
    elif args[0] == 'steer':
        variable, i = args[1][0], args[1][1]
        if variable == 'theta1':
            value = p.phi_max**2 - (theta[i] - theta1[i])**2
            
        elif variable == 'theta2':
            value = p.phi_max**2 - (theta[i] - theta2[i])**2
            
        return value
    
    #不等式制約として与えるomega,vの初期値
    elif args[0] == 'ini':
        variable, i = args[1][0], args[1][1]
        if variable == 'omega1':
            value = p.error_omega**2 - (omega1[0] - p.initial_omega1)**2
            
        elif variable == 'omega2':
            value = p.error_omega**2 - (omega2[0] - p.initial_omega2)**2
            
        elif variable == 'v1':
            value = p.error_v**2 - (v1[0] - p.initial_v1)**2
            
        elif variable == 'v2':
            value = p.error_v**2 - (v2[0] - p.initial_v2)**2
            
        return value
    
def generate_cons_with_jac():
    env_data = env.Env()
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    cons = ()
    
    #障害物回避のための不等式制約を追加する
    #矩形
    for k in range(len(obs_rectangle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['rectangle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
    #円形
    for k in range(len(obs_circle)):
        for i in range(p.N):
            args = ['avoid_obstacle', ['circle', k, i]]
            cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
            
            
            
    
    #運動学モデルの制約からなる等式制約を追加する
    #x1
    for i in range(p.N-1):
        args = ['model', ['x1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y1
    for i in range(p.N-1):
        args = ['model', ['y1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    #theta1
    for i in range(p.N-1):
        args = ['model', ['theta1', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x2
    for i in range(p.N-1):
        args = ['model', ['x2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y2
    for i in range(p.N-1):
        args = ['model', ['y2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    #theta2
    for i in range(p.N-1):
        args = ['model', ['theta2', i]]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        
        
    #境界値条件の等式制約を追加
    if p.set_cons['initial_x'] == False:
        pass
    else:
        args = ['boundary', ['x', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #x終端条件
    if p.set_cons['terminal_x'] == False:
        pass
    else:
        args = ['boundary', ['x', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)

    #y初期条件
    if p.set_cons['initial_y'] == False:
        pass
    else:
        args = ['boundary', ['y', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #y終端条件
    if p.set_cons['terminal_y'] == False:
        pass
    else:
        args = ['boundary', ['y', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta初期条件
    if p.set_cons['initial_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta終端条件
    if p.set_cons['terminal_theta'] == False:
        pass
    else:
        args = ['boundary', ['theta', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta1初期条件
    if p.set_cons['initial_theta1'] == False:
        pass
    else:
        args = ['boundary', ['theta1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta1終端条件
    if p.set_cons['terminal_theta1'] == False:
        pass
    else:
        args = ['boundary', ['theta1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta2初期条件
    if p.set_cons['initial_theta2'] == False:
        pass
    else:
        args = ['boundary', ['theta2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #theta2終端条件
    if p.set_cons['terminal_theta2'] == False:
        pass
    else:
        args = ['boundary', ['theta2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega1初期条件
    if p.set_cons['initial_omega1'] == False:
        pass
    else:
        args = ['boundary', ['omega1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega1終端条件
    if p.set_cons['terminal_omega1'] == False:
        pass
    else:
        args = ['boundary', ['omega1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega2初期条件
    if p.set_cons['initial_omega2'] == False:
        pass
    else:
        args = ['boundary', ['omega2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #omega2終端条件
    if p.set_cons['terminal_omega2'] == False:
        pass
    else:
        args = ['boundary', ['omega2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
        
    #v1初期条件
    if p.set_cons['initial_v1'] == False:
        pass
    else:
        args = ['boundary', ['v1', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v1終端条件
    if p.set_cons['terminal_v1'] == False:
        pass
    else:
        args = ['boundary', ['v1', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v2初期条件
    if p.set_cons['initial_v2'] == False:
        pass
    else:
        args = ['boundary', ['v2', 'ini']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #v2終端条件
    if p.set_cons['terminal_v2'] == False:
        pass
    else:
        args = ['boundary', ['v2', 'ter']]
        cons = cons + ({'type':'eq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    
    
    #ステアリング角の制約を加える
    for i in range(p.N):
        args = ['steer', ['theta1', i]]
        cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    for i in range(p.N):
        args = ['steer', ['theta2', i]]
        cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
        
    #不等式制約としてomega,vの初期値を与える
    args = ['ini', ['omega1', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    args = ['ini', ['omega2', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)    
    
    args = ['ini', ['v1', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)
    
    args = ['ini', ['v2', 0]]
    cons = cons + ({'type':'ineq', 'fun': constraint, 'jac': jac_of_constraint, 'args': args},)    
    
    return cons