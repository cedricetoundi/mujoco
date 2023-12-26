import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
#import control
import scipy as sp


xml_path = 'pendulum2.xml'
simend = 20

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def f(x,u):
    #x=q0,q1,qdot0,qdot1
    #u=torque

    data.qpos[0] = x[0]
    #data.qpos[1] = x[1]
    data.qvel[0] = x[1]
   # data.qvel[1] = x[3]
    data.ctrl[0] = u[0]
    mj.mj_forward(model,data) # mj_forward is continuous time dynamics f(t,x,u)

    #qddot = inv(M)*(data_ctrl-frc_bias)
    M = np.zeros((1,1))
    mj.mj_fullM(model,M,data.qM)
    invM = np.linalg.inv(M)
    frc_bias = np.array([data.qfrc_bias[0]])
    tau = np.array([u[0]])
    qddot = np.matmul(invM,np.subtract(tau,frc_bias))

    xdot = np.array([data.qvel[0],qddot[0]])
    return xdot


def linearize():

    

    n = 2
    m = 1
    A = np.zeros((n,n))
    B = np.zeros((n,m))
#     A[0][1]=1
#     A[2][3]=1
#     A[1][0]=(m1+m2)*g/(m1*l1)
#     A[3][2]=(m1+m2)*g/(m1*l2)
#     A[3][2]=-(m1+m2)*g/(m1*l2)
#     A[1][2]=-m2*g/(m1*l1)

#    B[1]=1/(m1*l1**2)
#    B[3]=-1/(m1*l1*l2)
#
    x0 = np.array([0,0])
    u0 = np.array([0])
    xdot0 = f(x0,u0)
    ##print(xdot0)

    pert = 1e-2
    ##get A matrix
    for i in range(0,n):
        x = [0]*n
        u = u0
        for j in range(0,n):
            x[j] = x0[j]
        x[i] = x[i]+pert
        xdot = f(x,u)
        for k in range(0,n):
            A[k,i] = (xdot[k]-xdot0[k])/pert

    ##get B matrix
    for i in range(0,m):
        x = x0
        u = [0]*m
        for j in range(0,m):
            u[j] = u0[j]
        u[i] = u[i]+pert
        xdot = f(x,u)
        for k in range(0,n):
            B[k,i] = (xdot[k]-xdot0[k])/pert

    return A,B


def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    #pass
    global K
    global R
    global B
    global P
    global g
    global D

    n = 2
    m = 1

    #1. linearization
    A,B = linearize()
    print(A)
    print(B)
    

    #2. linear quadratic regulator
    g=5
    Q = np.eye((n))
    R = np.block([[-g**2*np.eye((m)),np.zeros((m,m))],[np.zeros((m,m)),np.eye(m)]])
    D=np.array([1,0])
    B_new=np.append(B,D).reshape(2,2).T
    print(B_new)
    #K,S,E = control.lqr(A,B,Q,R)
    P=sp.linalg.solve_continuous_are(A,B_new,Q,R)
    print(P)
    # print("K = ",K)

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    # pass
    global K
    #global R
    global B
    global P
    global g
    global D

    #1. apply congtrol u = -K*x
    x = np.array([data.qpos[0],data.qvel[0]])
    #u = np.matmul(np.matmul(-np.linalg.inv(R),B.T),P).dot(x)
    u = ((-0.5*100*B.T) @ P) @ x
    
    data.ctrl[0] = u

    #2 apply disturbance torque
    #tau_disturb_mean = 0
    #tau_disturb_dev = 20
    #tau_d0 = np.random.normal(tau_disturb_mean,tau_disturb_dev)
    d=((1/(2*g^2)*D.T)@P)@x

    data.qfrc_applied[0] = d



def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
        # update button state
        button_left = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_middle = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(
            window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

#set initial conditions
data.qpos[0] = np.pi*3/2

# Set camera configuration
cam.azimuth = 90.0
cam.distance = 5.0
cam.elevation = -5
cam.lookat = np.array([0.012768, -0.000000, 1.254336])


# Initialize Controller

init_controller(model,data)
#set the controller
actuator_type = "servo"
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
