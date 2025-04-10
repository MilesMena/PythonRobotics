"""
In estimation theory, the extended Kalman filter (EKF) is the nonlinear version of the Kalman filter which linearizes about an estimate of the current mean and covariance. In the case of well defined transition models, the EKF has been considered the de facto standard in the theory of nonlinear state estimation, navigation systems and GPS.
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from utils.plot import plot_covariance_ellipse

class EKF:
    def __init__(self):
        self.Q = np.diag([
            0.1, # x-axis
            0.1,    # y-axis
            np.deg2rad(1.0), # yaw angle
            1.0 # velocity
        ]) **2
        self.R = np.diag([1.0, 1.0]) ** 2 # observation x,y position covariance

        # self.u = self.calc_input()
        #  Simulation parameter
        self.INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
        self.GPS_NOISE = np.diag([0.5, 0.5]) ** 2
        self.DT = 0.1  # time tick [s]
        self.SIM_TIME = 50.0  # simulation time [s]

    def calc_input(self):
        v = 1.0 #[m/s]
        yaw_rate = 0.1 # [rad/s]
        u = np.array([[v], [yaw_rate]])
        return u
    
    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
        B = np.array([[self.DT * math.cos(x[2, 0]), 0],
                  [self.DT * math.sin(x[2, 0]), 0],
                  [0.0, self.DT],
                  [1.0, 0.0]])
        x = F @ x + B @ u
        return x
    
    def jacob_f(self, x, u):
        yaw = x[2,0]
        v = u[0,0]
        jF = np.array([
        [1.0, 0.0, -self.DT * v * math.sin(yaw), self.DT * math.cos(yaw)],
        [0.0, 1.0, self.DT * v * math.cos(yaw), self.DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

        return jF
    
    def observation_model(self, x):
        H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
        z = H @ x
        return z
    
    def jacob_h(self):
        # Jacobian of Observation Model
        jH = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        return jH
            

    def observation(self,xTrue, xd, u):
        xTrue = self.motion_model(xTrue, u)
        z = self.observation_model(xTrue) + self.GPS_NOISE @np.random.randn(2,1)
        ud = u + self.INPUT_NOISE @ np.random.randn(2,1)
        xd = self.motion_model(xd, ud)
        return xTrue, z, xd, ud
    
    def ekf_estimation(self, xEst, PEst, z, u):
        xPred = self.motion_model(xEst, u)
        jF = self.jacob_f(xEst, u)
        PPred = jF @ PEst @ jF.T + self.Q

        jH = self.jacob_h()
        zPred = self.observation_model(xPred)
        y = z - zPred
        S = jH @ PPred @ jH.T + self.R
        K = PPred @ jH.T @ np.linalg.inv(S)
        xEst = xPred + K @ y
        PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
        return xEst, PEst
    
    def run(self):
        time = 0.0

        xEst = np.zeros((4,1))
        xTrue = np.zeros((4,1))
        PEst = np.eye(4)

        xDR = np.zeros((4,1)) # dead reckoning
        
        # history
        hxEst = xEst
        hxTrue = xTrue
        hxDR = xTrue
        hz = np.zeros((2,1))

        while self.SIM_TIME >= time:
            time += self.DT
            u = self.calc_input()

            xTrue, z , xDR, ud = self.observation(xTrue, xDR, u)
            xEst, PEst = self.ekf_estimation(xEst, PEst, z, ud)
            # store data history
            hxEst = np.hstack((hxEst, xEst))
            hxDR = np.hstack((hxDR, xDR))
            hxTrue = np.hstack((hxTrue, xTrue))
            hz = np.hstack((hz, z))


            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == "__main__":
    ekf = EKF()
    ekf.run()

