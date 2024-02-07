import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import env
from param import Parameter as p
import csv
import numpy as np
import util
from scipy.interpolate import Akima1DInterpolator, interp1d
import os

def path_interpolation(xs1, ys1, thetas1, xs2, ys2, thetas2, point=100):
    t = np.linspace(0, 1, len(xs1))
            
    fx1 = interp1d(t, xs1)
    fy1 = interp1d(t, ys1)
    ftheta1 = interp1d(t, thetas1)
    
    fx2 = interp1d(t, xs2)
    fy2 = interp1d(t, ys2)
    ftheta2 = interp1d(t, thetas2)
        
    t = np.linspace(0, 1, point)

    return fx1(t), fy1(t), ftheta1(t), fx2(t), fy2(t), ftheta2(t) 

def plot_path(ax, x1, y1, theta1, x2, y2, theta2):
    print(len(x2), len(y2))
    ax.scatter(x1, y1, marker='x', color='red', s=5)
    ax.scatter(x2, y2, marker='x', color='blue', s=5)
    
    env_data = env.Env()
    wall_list = env_data.obs_boundary
    obs_rectangle = env_data.obs_rectangle
    obs_circle = env_data.obs_circle
    
    #wallを配置
    for k in range(len(wall_list)):
        wall = patches.Rectangle((wall_list[k][0], wall_list[k][1]), wall_list[k][2], wall_list[k][3], linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(wall)
    
    #障害物を配置
    for k in range(len(obs_rectangle)):
        x0, y0, w, h = obs_rectangle[k][0], obs_rectangle[k][1], obs_rectangle[k][2], obs_rectangle[k][3]
        rectangle_obstacle = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rectangle_obstacle)
        
    for k in range(len(obs_circle)):
        x_o, y_o, r_o = obs_circle[k][0], obs_circle[k][1], obs_circle[k][2],
        circle_obstacle = patches.Circle((x_o, y_o), radius=r_o, edgecolor='black', facecolor='gray')
        ax.add_patch(circle_obstacle)
    
    ax.set_xlabel(r'$x$[m]')
    ax.set_ylabel(r'$y$[m]')
    ax.set_xlim([p.x_min - p.margin, p.x_max + p.margin])
    ax.set_ylim([p.y_min - p.margin, p.y_max + p.margin])
    
    """
    #startとgoalを配置
    ax.scatter([start[0]], [start[1]], marker='v', color='green', label='start')
    ax.scatter([goal[0]], [goal[1]], marker='^', color='green', label='goal')
    """
    ax.quiver(x1[-1], y1[-1], np.cos(theta1[-1]), np.sin(theta1[-1]))
    ax.quiver(x2[-1], y2[-1], np.cos(theta2[-1]+np.pi), np.sin(theta2[-1]+np.pi))
    
    ax.set_aspect('equal')
    


def gen_movie(xs1, ys1, thetas1, xs2, ys2, thetas2):
    fig = plt.figure()
    frames = []
    
    xs1, ys1, thetas1, xs2, ys2, thetas2 = path_interpolation(xs1, ys1, thetas1, xs2, ys2, thetas2, 100)
    for i in range(1, len(xs1) + 1):
        ax = plt.axes()
        x1, y1, theta1, x2, y2, theta2 = xs1[:i], ys1[:i], thetas1[:i], xs2[:i], ys2[:i], thetas2[:i]
        plot_path(ax, x1, y1, theta1, x2, y2, theta2)
        frames.append([ax])
    ani = animation.ArtistAnimation(fig=fig, artists=frames, interval=100)
    ani.save("video/plot_path_11.mp4")

def plot_rotated_rectangle(ax, center, width, height, angle):
    # 長方形の中心から左上の座標を計算
    x = center[0] - 0.5 * width * np.cos(angle) + 0.5 * height * np.sin(angle)
    y = center[1] - 0.5 * width * np.sin(angle) - 0.5 * height * np.cos(angle)

    # 長方形を描画
    rectangle = patches.Rectangle((x, y), width, height, angle=np.rad2deg(angle), edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)
    
def plot_robot(ax, x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, xs, ys, vis_v):
    ax.plot(xs, ys, linestyle='--')
    
    wheel_r = 0.075
    wheel_hwidth = 0.03
    robot_half_width = 0.175
    wheel_shaft_width = 0.02
    
    arm_length = 0.25
    arm_hwidth = 0.02
    
    cart_hwidth = 0.4
    cart_width = 0.6
    plot_rotated_rectangle(ax, (x, y), cart_width, cart_hwidth, theta)
    
    box_hwidth = 0.35
    box_width = 0.35
    plot_rotated_rectangle(ax, (x, y), box_width, box_hwidth, theta)
    
    board_hwidth = 0.25
    board_width = 0.04
    
    body_hwidth = robot_half_width * 2
    body_width = 0.175*2
    
    #folloer
    #wheel
    wheel_l_x1, wheel_l_y1 = x1 + robot_half_width*np.cos(theta1+np.pi/2), y1 + robot_half_width*np.sin(theta1+np.pi/2)
    plot_rotated_rectangle(ax, (wheel_l_x1, wheel_l_y1), wheel_r*2, wheel_hwidth, theta1)
    
    wheel_r_x1, wheel_r_y1 = x1 + robot_half_width*np.cos(theta1-np.pi/2), y1 + robot_half_width*np.sin(theta1-np.pi/2)
    plot_rotated_rectangle(ax, (wheel_r_x1, wheel_r_y1), wheel_r*2, wheel_hwidth, theta1)
    
    #wheel_ax
    plot_rotated_rectangle(ax, (x1, y1), wheel_shaft_width, robot_half_width*2, theta1)
    
    #arm
    arm_x1, arm_y1 = x1 + arm_length/2*np.cos(theta), y1 + arm_length/2*np.sin(theta)
    plot_rotated_rectangle(ax, (arm_x1, arm_y1), arm_length, arm_hwidth, theta)
    
    #board
    board_x1, board_y1 = x1 + (arm_length-board_width/2)*np.cos(theta), y1 + (arm_length-board_width/2)*np.sin(theta)
    plot_rotated_rectangle(ax, (board_x1, board_y1), board_width, board_hwidth, theta)
    
    #body_arc
    arc_x1, arc_y1 = x1, y1
    arc1 = patches.Arc((x1, y1), robot_half_width*2, robot_half_width*2, angle=np.rad2deg(theta1), theta1=-90, theta2=90, edgecolor="red")
    ax.add_patch(arc1)
    
    body_x1, body_y1 = x1 - body_width/2*np.cos(theta1), y1 - body_width/2*np.sin(theta1)
    plot_rotated_rectangle(ax, (body_x1, body_y1), body_width, body_hwidth, theta1)

    #leader
    #wheel
    wheel_l_x2, wheel_l_y2 = x2 + robot_half_width*np.cos(theta2+np.pi/2), y2 + robot_half_width*np.sin(theta2+np.pi/2)
    plot_rotated_rectangle(ax, (wheel_l_x2, wheel_l_y2), wheel_r*2, wheel_hwidth, theta2)
    
    wheel_r_x2, wheel_r_y2 = x2 + robot_half_width*np.cos(theta2-np.pi/2), y2 + robot_half_width*np.sin(theta2-np.pi/2)
    plot_rotated_rectangle(ax, (wheel_r_x2, wheel_r_y2), wheel_r*2, wheel_hwidth, theta2)
    
    #wheel_ax
    plot_rotated_rectangle(ax, (x2, y2), wheel_shaft_width, robot_half_width*2, theta2)
    
    #arm
    arm_x2, arm_y2 = x2 - arm_length/2*np.cos(theta), y2 - arm_length/2*np.sin(theta)
    plot_rotated_rectangle(ax, (arm_x2, arm_y2), arm_length, arm_hwidth, theta)
    
    #board
    board_x2, board_y2 = x2 - (arm_length-board_width/2)*np.cos(theta), y2 - (arm_length-board_width/2)*np.sin(theta)
    plot_rotated_rectangle(ax, (board_x2, board_y2), board_width, board_hwidth, theta)
    
    #body_arc
    arc_x2, arc_y2 = x2, y2
    arc2 = patches.Arc((x2, y2), robot_half_width*2, robot_half_width*2, angle=np.rad2deg(theta2), theta1=90, theta2=270, edgecolor="red")
    ax.add_patch(arc2)
    
    body_x2, body_y2 = x2 + body_width/2*np.cos(theta2), y2 + body_width/2*np.sin(theta2)
    plot_rotated_rectangle(ax, (body_x2, body_y2), body_width, body_hwidth, theta2)
    
    if vis_v == True:
        ax.quiver(x1, y1, v1*np.cos(theta1), v1*np.sin(theta1), angles='xy', scale_units='xy', scale=1)
        ax.quiver(x2, y2, v2*np.cos(theta2), v2*np.sin(theta2), angles='xy', scale_units='xy', scale=1)
        
    # プロットの設定
    
    ax.set_xlim(x-(cart_width/2 + arm_length + body_width + 0.2), x + cart_width/2 + arm_length + body_width + 0.2)
    ax.set_ylim(y-(cart_width/2 + arm_length + body_width + 0.2), y + cart_width/2 + arm_length + body_width + 0.2)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    # x軸に補助目盛線を設定
    ax.grid(which = "both", axis = "x", color = "black", alpha = 0.8,
            linestyle = "--", linewidth = 0.5)

    # y軸に目盛線を設定
    ax.grid(which = "both", axis = "y", color = "black", alpha = 0.8,
            linestyle = "--", linewidth = 0.5)
    """
    ax.set_xlim(-3, 33)
    ax.set_ylim(-10, 10)
    """
    ax.set_aspect('equal', adjustable='box')  # アスペクト比を保持
    
def interpolation(x, points=100, is_spline=True):
    t = np.linspace(0, 1, len(x))      
    if is_spline == True:
        fx = Akima1DInterpolator(t, x)
    else:
        fx = interp1d(t, x)
        
    t = np.linspace(0, 1, points)

    return fx(t)

def plot_robot_path(ax, x1_path, y1_path, x2_path, y2_path):
    ax.plot(x1_path, y1_path, linestyle='--')
    ax.plot(x2_path, y2_path, linestyle='--')
    
def gen_robot_movie(xs, ys, thetas, thetas1, thetas2, omegas1, omegas2, vs1, vs2, xs1, ys1, xs2, ys2, is_interpolation=False, vis_v=False, vis_robot_path=False):
    if is_interpolation == True:
        xs, ys, thetas, thetas1, thetas2, omegas1, omegas2, vs1, vs2, xs1, ys1, xs2, ys2 = interpolation(xs), interpolation(ys),interpolation(thetas),interpolation(thetas1),interpolation(thetas2),interpolation(omegas1),interpolation(omegas2),interpolation(vs1),interpolation(vs2),interpolation(xs1),interpolation(ys1),interpolation(xs2),interpolation(ys2)
    fig = plt.figure()
    frames = []
    for i in range(len(xs)):
        #座標を(0, 0)にずらさない
        x, y, x1, y1, x2, y2 = xs[i], ys[i], xs1[i], ys1[i], xs2[i], ys2[i]
        """
        #座標(0, 0)にずらす
        x, y, x1, y1, x2, y2 = xs[i] - xs[i], ys[i] - ys[i], xs1[i] - xs[i], ys1[i] - ys[i], xs2[i] - xs[i], ys2[i] - ys[i] 
        """
        theta, theta1, theta2, omega1, omega2, v1, v2 = thetas[i], thetas1[i], thetas2[i], omegas1[i], omegas2[i], vs1[i], vs2[i]
        ax = plt.axes()
        plot_robot(ax, x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, xs, ys, vis_v)
        if vis_robot_path == True:
            x1_path, y1_path, x2_path, y2_path = xs1[:i+1], ys1[:i+1], xs2[:i+1], ys2[:i+1], 
            plot_robot_path(ax, x1_path, y1_path, x2_path, y2_path)
        frames.append([ax])
    ani = animation.ArtistAnimation(fig=fig, artists=frames)
    ani.save("video/plot_path.mp4")
    
def gen_robot_fig(xs, ys, thetas, thetas1, thetas2, omegas1, omegas2, vs1, vs2, xs1, ys1, xs2, ys2, is_interpolation=False, vis_v=False, vis_robot_path=False):
    if is_interpolation == True:
        xs, ys, thetas, thetas1, thetas2, omegas1, omegas2, vs1, vs2, xs1, ys1, xs2, ys2 = interpolation(xs), interpolation(ys),interpolation(thetas),interpolation(thetas1),interpolation(thetas2),interpolation(omegas1),interpolation(omegas2),interpolation(vs1),interpolation(vs2),interpolation(xs1),interpolation(ys1),interpolation(xs2),interpolation(ys2)
    fig = plt.figure()
    counter = 0
    dirname = "path_fig/"
    os.makedirs(dirname, exist_ok=True)
    for i in range(len(xs)):
        #座標を(0, 0)にずらさない
        x, y, x1, y1, x2, y2 = xs[i], ys[i], xs1[i], ys1[i], xs2[i], ys2[i]
        """
        #座標(0, 0)にずらす
        x, y, x1, y1, x2, y2 = xs[i] - xs[i], ys[i] - ys[i], xs1[i] - xs[i], ys1[i] - ys[i], xs2[i] - xs[i], ys2[i] - ys[i] 
        """
        theta, theta1, theta2, omega1, omega2, v1, v2 = thetas[i], thetas1[i], thetas2[i], omegas1[i], omegas2[i], vs1[i], vs2[i]
        ax = plt.axes()
        plot_robot(ax, x, y, theta, theta1, theta2, omega1, omega2, v1, v2, x1, y1, x2, y2, xs, ys, vis_v)
        if vis_robot_path == True:
            x1_path, y1_path, x2_path, y2_path = xs1[:i+1], ys1[:i+1], xs2[:i+1], ys2[:i+1], 
            plot_robot_path(ax, x1_path, y1_path, x2_path, y2_path)
        filename = dirname +"path_fig_{}.png".format(counter)
        fig.savefig(filename)
        counter += 1
        plt.clf()

        
if __name__ == '__main__':
    #csvからnetwork情報を取得
    with open("network_circle.csv") as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        trajectory_vectors = [row for row in reader]
        
    path_index = 5
    trajectory_vector = trajectory_vectors[path_index]
    trajectory_vector = np.array(trajectory_vector)
    trajectory_matrix = util.vector_to_matrix(trajectory_vector)
    xs, ys = trajectory_matrix[0], trajectory_matrix[1]
    gen_movie(xs, ys)