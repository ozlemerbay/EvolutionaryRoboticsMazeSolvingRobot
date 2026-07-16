import pybullet as p
import numpy as np
import math

class RobotInterface:
    def __init__(self, robot_id, sensor_range):
        """init robot"""
        self.robot_id = robot_id
        self.sensor_range = sensor_range
        self.sensor_angles = [0, math.radians(45), math.radians(-45)]

    def get_sensor_data(self):
        """calculate and return distances from sensors"""
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        # convert orient to matrix so I can get position vectors
        # matrix columns are forward, left and right
        # matrix rows are x,y and z
        rot_matrix = p.getMatrixFromQuaternion(orient)

        forward = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
        # calculate (-y,x) from forward to make it left to save time
        left = np.array([-rot_matrix[3], rot_matrix[0], rot_matrix[6]])

        readings = []
        for angle in self.sensor_angles:
            cos_angle, sin_angle = math.cos(angle), math.sin(angle)
            sensor_direction = np.array([
                cos_angle * forward[0] + sin_angle * left[0],
                cos_angle * forward[1] + sin_angle * left[1],
                0
            ])
            sensor_end_pos = pos + sensor_direction * self.sensor_range

            result = p.rayTest(pos, sensor_end_pos)[0]
            readings.append(result[2]) # only get the distance percentage robot can move from result

        return readings

    def set_motor_velocities(self, left_speed, right_speed):
        """send command to wheels."""
        # left wheel
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_speed
        )
        # right wheel
        p.setJointMotorControl2(
            bodyIndex=self.robot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_speed
        )