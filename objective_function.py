from param import Parameter as p
import numpy as np
import util



def objective_function(x, *args):
    #matrixに変換
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N):
        sum += (theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + (theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + (omega1[i] ** 2 / p.omega1_max ** 2) + (omega2[i] ** 2 / p.omega2_max ** 2) + 0*(v1[i] ** 2 / p.v1_max ** 2) + 0*(v2[i] ** 2 / p.v2_max ** 2)
    
    return sum*dt / p.N


def jac_of_objective_function(x, *args):
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0

    for i in range(p.N):
        jac_theta[i] = 2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta_max*dt + 2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)/p.theta_max*dt
        jac_theta1[i] = -2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta1_max*dt
        jac_theta2[i] = -2*(theta[i]/p.theta_max - theta2[i]/p.theta1_max)/p.theta2_max*dt
        
        #phiの微分
        jac_omega1[i] = (omega1[i] * 2) / ((p.omega1_max ** 2))*dt
        
        jac_omega2[i] = (omega2[i] * 2) / ((p.omega2_max ** 2))*dt
    
        #vの微分
        jac_v1[i] = 0*(v1[i] * 2) / ((p.v1_max ** 2))*dt
        
        jac_v2[i] = 0*(v2[i] * 2) / ((p.v2_max ** 2))*dt
        
    #dtの微分
    #phiの二乗和を目的関数とする。
    for i in range(p.N):
        jac_dt += (theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + (theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + 0*(omega1[i] ** 2 / p.omega1_max ** 2) + 0*(omega2[i] ** 2 / p.omega2_max ** 2) + 0*(v1[i] ** 2 / p.v1_max ** 2) + 0*(v2[i] ** 2 / p.v2_max ** 2)

    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons/p.N

def sigmoid(x, a = 100):
    return 1 / (1 + np.exp(-a*x))


def grad_sigmoid(x, a = 100):
    return a*np.exp(-a*x) / (1 + np.exp(-a*x))**2


def objective_function2(x, *args):
    w1, w2 = args[0], args[1]
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    

    sum = 0
    for i in range(p.N):
        sum += (trajectory_matrix[3, i] ** 2 / p.phi_max ** 2) + (trajectory_matrix[4, i] ** 2 / p.v_max ** 2) + w1*sigmoid(trajectory_matrix[4, i])*(trajectory_matrix[4, i] ** 2 / p.v_max ** 2) + w2*sigmoid(-trajectory_matrix[4, i])*(trajectory_matrix[4, i] ** 2 / p.v_max ** 2)
    
    return sum / p.N


def jac_of_objective_function2(x, *args):
    w1, w2 = args[0], args[1]
    #matrixに変換
    trajectory_matrix = x.reshape(p.M, p.N)
    
    jac_f = np.zeros((p.M, p.N))

    for i in range(p.N):
        #phiの微分
        jac_f[3, i] = (trajectory_matrix[3, i] * 2) / (p.N * (p.phi_max ** 2))  
    
        #vの微分
        jac_f[4, i] = (trajectory_matrix[4, i] * 2) / (p.N * (p.v_max ** 2))  + w1*(grad_sigmoid(trajectory_matrix[4, i])*(trajectory_matrix[4, i] ** 2 / p.v_max ** 2) + sigmoid(trajectory_matrix[4, i])*(trajectory_matrix[4, i] * 2) / (p.N * (p.v_max ** 2))) + w2*(-grad_sigmoid(-trajectory_matrix[4, i])*(trajectory_matrix[4, i] ** 2 / p.v_max ** 2) + sigmoid(-trajectory_matrix[4, i])*(trajectory_matrix[4, i] * 2) / (p.N * (p.v_max ** 2)))

    #ベクトルに直す
    jac_f = jac_f.flatten()
    
    return jac_f


def objective_function3(x, *args):
    #matrixに変換
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    return dt

def jac_of_objective_function3(x, *args): 
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 1

    
    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons


def objective_function4(x, *args):
    #matrixに変換
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    #phiの二乗和を目的関数とする。
    sum = 0
    for i in range(p.N-1):
        sum += (theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + (theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + (omega1[i] ** 2 / p.omega1_max ** 2) + (omega2[i] ** 2 / p.omega2_max ** 2) + ((v1[i+1] - v1[i])**2 / p.v1_max ** 2) + ((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)
    
    return sum*dt / p.N

def jac_of_objective_function4(x, *args): 
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0

    for i in range(p.N-1):
        jac_theta[i] = 2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta_max*dt + 2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)/p.theta_max*dt
        jac_theta1[i] = -2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta1_max*dt
        jac_theta2[i] = -2*(theta[i]/p.theta_max - theta2[i]/p.theta1_max)/p.theta2_max*dt
        
        jac_omega1[i] = (omega1[i] * 2) / ((p.omega1_max ** 2))*dt
        
        jac_omega2[i] = (omega2[i] * 2) / ((p.omega2_max ** 2))*dt
    
    #vの微分
    jac_v1[0] = -((v1[1] - v1[0]) * 2) / ((p.v1_max ** 2))*dt
    jac_v1[p.N-1] = ((v1[p.N-1] - v1[p.N-2]) * 2) / ((p.v1_max ** 2))*dt 
    
    jac_v2[0] = -((v2[1] - v2[0]) * 2) / ((p.v2_max ** 2))*dt
    jac_v2[p.N-1] = ((v2[p.N-1] - v2[p.N-2]) * 2) / ((p.v2_max ** 2))*dt 
    
    for i in range(1, p.N-1):
        jac_v1[i] = ((v1[i] - v1[i-1]) * 2) / ((p.v1_max ** 2))*dt - ((v1[i+1] - v1[i]) * 2) / ((p.v1_max ** 2))*dt 
        jac_v2[i] = ((v2[i] - v2[i-1]) * 2) / ((p.v2_max ** 2))*dt - ((v2[i+1] - v2[i]) * 2) / ((p.v2_max ** 2))*dt 
        
    #dtの微分
    for i in range(p.N-1):
        jac_dt += (theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + (theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + (omega1[i] ** 2 / p.omega1_max ** 2) + (omega2[i] ** 2 / p.omega2_max ** 2) + ((v1[i+1] - v1[i])**2 / p.v1_max ** 2) + ((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)

    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons/p.N


#論文実験用評価関数1
def objective_function_ex1(x, *args):
    w1, w2, w3 = args[0], args[1], args[2]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    sum = 0
    
    for i in range(p.N-1):
        sum += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt + w1*((v1[i+1] - v1[i])**2 / p.v1_max ** 2)*dt + w1*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)*dt + w2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt + w3*dt
    
    return sum / p.N
    
def jac_of_objective_function_ex1(x, *args):
    w1, w2, w3 = args[0], args[1], args[2]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0
    
    for i in range(p.N-1):
        jac_theta[i] = w2*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta_max*dt + w2*2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)/p.theta_max*dt
        jac_theta1[i] = -w2*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta1_max*dt
        jac_theta2[i] = -w2*2*(theta[i]/p.theta_max - theta2[i]/p.theta1_max)/p.theta2_max*dt
        
        jac_omega1[i] = w1*(omega1[i] * 2) / ((p.omega1_max ** 2))*dt
        
        jac_omega2[i] = w1*(omega2[i] * 2) / ((p.omega2_max ** 2))*dt
    
    #vの微分
    jac_v1[0] = -w1*((v1[1] - v1[0]) * 2) / ((p.v1_max ** 2))*dt
    jac_v1[p.N-1] = w1*((v1[p.N-1] - v1[p.N-2]) * 2) / ((p.v1_max ** 2))*dt 
    
    jac_v2[0] = -w1*((v2[1] - v2[0]) * 2) / ((p.v2_max ** 2))*dt
    jac_v2[p.N-1] = w1*((v2[p.N-1] - v2[p.N-2]) * 2) / ((p.v2_max ** 2))*dt 
    
    for i in range(1, p.N-1):
        jac_v1[i] = w1*((v1[i] - v1[i-1]) * 2) / ((p.v1_max ** 2))*dt - w1*((v1[i+1] - v1[i]) * 2) / ((p.v1_max ** 2))*dt 
        jac_v2[i] = w1*((v2[i] - v2[i-1]) * 2) / ((p.v2_max ** 2))*dt - w1*((v2[i+1] - v2[i]) * 2) / ((p.v2_max ** 2))*dt 
        
    #dtの微分
    for i in range(p.N-1):
        jac_dt += w2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + w2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + w1*(omega1[i] ** 2 / p.omega1_max ** 2) + w1*(omega2[i] ** 2 / p.omega2_max ** 2) + w1*((v1[i+1] - v1[i])**2 / p.v1_max ** 2) + w1*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2) + w3

    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons/p.N


#論文実験用評価関数2
def objective_function_ex2(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    sum = 0
    
    for i in range(p.N-1):
        sum += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt + w2*((v1[i+1] - v1[i])**2 / p.v1_max ** 2)*dt + w2*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)*dt + w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt + w4*dt
    
    return sum
    
def jac_of_objective_function_ex2(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0
    
    for i in range(p.N-1):
        jac_theta[i] = w3*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta_max*dt + w3*2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)/p.theta_max*dt
        jac_theta1[i] = -w3*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta1_max*dt
        jac_theta2[i] = -w3*2*(theta[i]/p.theta_max - theta2[i]/p.theta1_max)/p.theta2_max*dt
        
        jac_omega1[i] = w1*(omega1[i] * 2) / ((p.omega1_max ** 2))*dt
        
        jac_omega2[i] = w1*(omega2[i] * 2) / ((p.omega2_max ** 2))*dt
    
    #vの微分
    jac_v1[0] = -w2*((v1[1] - v1[0]) * 2) / ((p.v1_max ** 2))*dt
    jac_v1[p.N-1] = w2*((v1[p.N-1] - v1[p.N-2]) * 2) / ((p.v1_max ** 2))*dt 
    
    jac_v2[0] = -w2*((v2[1] - v2[0]) * 2) / ((p.v2_max ** 2))*dt
    jac_v2[p.N-1] = w2*((v2[p.N-1] - v2[p.N-2]) * 2) / ((p.v2_max ** 2))*dt 
    
    for i in range(1, p.N-1):
        jac_v1[i] = w2*((v1[i] - v1[i-1]) * 2) / ((p.v1_max ** 2))*dt - w2*((v1[i+1] - v1[i]) * 2) / ((p.v1_max ** 2))*dt 
        jac_v2[i] = w2*((v2[i] - v2[i-1]) * 2) / ((p.v2_max ** 2))*dt - w2*((v2[i+1] - v2[i]) * 2) / ((p.v2_max ** 2))*dt 
        
    #dtの微分
    for i in range(p.N-1):
        jac_dt += w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + w1*(omega1[i] ** 2 / p.omega1_max ** 2) + w1*(omega2[i] ** 2 / p.omega2_max ** 2) + w2*((v1[i+1] - v1[i])**2 / p.v1_max ** 2) + w2*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2) + w4

    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons

#論文実験用評価関数2
def evaluation_ex2(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    sum = 0
    J1, J2, J3, J4 = 0, 0, 0, 0
    for i in range(p.N-1):
        J1 += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt
        J2 += w2*((v1[i+1] - v1[i])**2 / p.v1_max ** 2)*dt + w2*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)*dt
        J3 += w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt
        J4 += w4*dt
        sum += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt + w2*((v1[i+1] - v1[i])**2 / p.v1_max ** 2)*dt + w2*((v2[i+1] - v2[i]) ** 2 / p.v2_max ** 2)*dt + w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt + w4*dt
    
    return J1, J2, J3, J4


#論文実験用評価関数3
def objective_function_ex3(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    sum = 0
    
    for i in range(p.N-1):
        sum += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt + w2*(v1[i] ** 2 / p.v1_max ** 2)*dt + w2*(v2[i] ** 2 / p.v2_max ** 2)*dt + w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt + w4*dt
    
    return sum 
    
def jac_of_objective_function_ex3(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    jac_cons = np.zeros((p.M, p.N))
    jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2 = jac_cons[0], jac_cons[1], jac_cons[2], jac_cons[3], jac_cons[4], jac_cons[5], jac_cons[6], jac_cons[7], jac_cons[8]
    jac_dt = 0
    
    for i in range(p.N-1):
        jac_theta[i] = w3*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta_max*dt + w3*2*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)/p.theta_max*dt
        jac_theta1[i] = -w3*2*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)/p.theta1_max*dt
        jac_theta2[i] = -w3*2*(theta[i]/p.theta_max - theta2[i]/p.theta1_max)/p.theta2_max*dt
        
        jac_omega1[i] = w1*(omega1[i] * 2) / ((p.omega1_max ** 2))*dt
        
        jac_omega2[i] = w1*(omega2[i] * 2) / ((p.omega2_max ** 2))*dt
        
        jac_v1[i] = w2*(v1[i] * 2) / ((p.v1_max ** 2))*dt
        
        jac_v2[i] = w2*(v2[i] * 2) / ((p.v2_max ** 2))*dt
    
     
    #dtの微分
    for i in range(p.N-1):
        jac_dt += w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2 + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2 + w1*(omega1[i] ** 2 / p.omega1_max ** 2) + w1*(omega2[i] ** 2 / p.omega2_max ** 2) + w2*(v1[i] ** 2 / p.v1_max ** 2) + w2*(v2[i] ** 2 / p.v2_max ** 2) + w4

    #ベクトルに直す
    jac_cons = np.hstack([jac_x, jac_y, jac_theta, jac_theta1, jac_theta2, jac_omega1, jac_omega2, jac_v1, jac_v2, jac_dt])
    
    return jac_cons


#論文実験用評価関数3
def evaluation_ex3(x, *args):
    w1, w2, w3, w4 = args[0], args[1], args[2], args[3]
    
    x, y, theta, theta1, theta2, omega1, omega2, v1, v2, dt = util.generate_result(x)
    
    sum = 0
    J1, J2, J3, J4 = 0, 0, 0, 0
    for i in range(p.N-1):
        J1 += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt
        J2 += w2*(v1[i] ** 2 / p.v1_max ** 2)*dt + w2*(v2[i] ** 2 / p.v2_max ** 2)*dt
        J3 += w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt
        J4 += w4*dt
        sum += w1*(omega1[i] ** 2 / p.omega1_max ** 2)*dt + w1*(omega2[i] ** 2 / p.omega2_max ** 2)*dt + w2*(v1[i] ** 2 / p.v1_max ** 2)*dt + w2*(v2[i] ** 2 / p.v2_max ** 2)*dt + w3*(theta[i]/p.theta_max - theta1[i]/p.theta1_max)**2*dt + w3*(theta[i]/p.theta_max - theta2[i]/p.theta2_max)**2*dt + w4*dt
    
    return J1, J2, J3, J4 