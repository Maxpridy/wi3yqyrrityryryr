import os
import mujoco_py
import numpy as np
from gym.utils import seeding
import time

class JacoEnv():
    def __init__(self,
                 width,
                 height,
                 frame_skip):
        self.frame_skip = frame_skip
        self.width = width
        self.height = height

        # Instantiate Mujoco model
        model_path = "jaco.xml"
        fullpath = os.path.join(
            os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model)

        self.init_state = self.sim.get_state()
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Setup actuators
        self.actuator_bounds = self.sim.model.actuator_ctrlrange
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_ctrlrange = self.actuator_high - self.actuator_low
        self.num_actuators = len(self.actuator_low)

        # init model_data_ctrl
        self.null_action = np.zeros(self.num_actuators)
        self.sim.data.ctrl[:] = self.null_action

        self.seed()

        self.sum_reward = 0
        self.rewarding_distance = 0.5

        # Target position bounds
        self.target_bounds = np.array(((-0.4, 0.6), (0.1, -0.3), (0.2, 0.3)))
        #self.goal = np.array((0.6, 0.3, 0.3))
        self.goal = self.sim.data.get_body_xpos("target")
        self.target_reset_distance = 0.2

        self.prev_centerd = 1

        #self.reset_target()


        # Setup discrete action space
        #self.control_values = self.actuator_ctrlrange * control_magnitude

        #self.num_actions = 5
        # self.action_space = [list(range(self.num_actions))
        #                      ] * self.num_actuators
        # self.observation_space = ((0, ), (height, width, 3),
        #                           (height, width, 3))

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_qpos_qvel(self, qpos, qvel):
        assert qpos.shape == (self.sim.model.nq, ) and qvel.shape == (
            self.sim.model.nv, )
        self.sim.data.qpos[:] = qpos
        self.sim.data.qvel[:] = qvel
        self.sim.forward()

    def reset(self):
        # Random initial position of Jaco
        # qpos = self.init_qpos + np.random.randn(self.sim.nv)

        #  Fixed initial position of Jaco
        qpos = self.init_qpos
        qvel = self.init_qvel

        # random object position start of episode
        #self.reset_target()

        # set initial joint positions and velocities
        self.set_qpos_qvel(qpos, qvel)

        return self.get_obs()

    def reset_target(self):
        #Randomize goal position within specified bounds

        # self.goal = np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]
        # self.goal = 1 * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]
        # geom_positions = self.sim.model.geom_pos.copy()
        # prev_goal_location = geom_positions[1]
        #
        # while (np.linalg.norm(prev_goal_location - self.goal) < self.target_reset_distance):
        #     #self.goal = np.random.rand(3) * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]
        #     self.goal = 1 * (self.target_bounds[:, 1] - self.target_bounds[:, 0]) + self.target_bounds[:, 0]

        #geom_positions[1] = self.goal
        #self.sim.model.geom_pos[:] = geom_positions
        while(True):
            pos_first = self.sim.model.body_pos[-1][0] * np.random.randn() * 2
            if pos_first > 0.3 and pos_first < 0.6:
                break
        while(True):
            pos_second = self.sim.model.body_pos[-1][1] * np.random.randn() * 2
            if pos_second > 0.3 and pos_second < 0.6:
                break
        while(True):
            pos_third = self.sim.model.body_pos[-1][2] * np.random.randn() * 2
            if pos_third > 0.3 and pos_third < 0.6:
                break

        self.sim.model.body_pos[-1] = [pos_first, pos_second, pos_third]



    def render(self, camera_name=None):
        rgb = self.sim.render(
            width=self.width, height=self.height, camera_name=camera_name)
        return rgb

    def _get_obs_joint(self):
        return np.concatenate(
            [self.sim.data.qpos.flat[:9], self.sim.data.qvel.flat[:9]])

    def _get_obs_rgb_view1(self):
        obs_rgb_view1 = self.render(camera_name='view1')
        return obs_rgb_view1

    def _get_obs_rgb_view2(self):
        obs_rgb_view2 = self.render(camera_name='view2')
        return obs_rgb_view2

    def get_obs(self):
        # return (self._get_obs_joint(), self._get_obs_rgb_view1(),
        #         self._get_obs_rgb_view2())
        #return self._get_obs_joint()
        return np.concatenate((self._get_obs_joint(), self.sim.data.get_body_xpos("target")))

    def do_simulation(self, ctrl):
        '''Do one step of simulation, taking new control as target

        Arguments:
            ctrl {np.array(num_actuator)}  -- new control to send to actuators
        '''
        ctrl = np.min((ctrl, self.actuator_high), axis=0)
        ctrl = np.max((ctrl, self.actuator_low), axis=0)

        self.sim.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            self.sim.step()

    def reduce_rewarding_distance(self, ep):
        self.rewarding_distance = ep

    # @profile(immediate=True)
    def step(self, a):
        dist = np.zeros(3)
        done = False
        new_control = np.copy(a).flatten()

        # Compute reward:
        # If any finger is close enough to target => +1
        dist[0] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_1") - self.sim.data.get_body_xpos("target"))
        dist[1] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_2") - self.sim.data.get_body_xpos("target"))
        dist[2] = np.linalg.norm(
            self.sim.data.get_body_xpos("jaco_link_finger_3") - self.sim.data.get_body_xpos("target"))

        # if continuous reward
        # reward = float((np.mean(dist)**-1)*0.1)
        reward = 0

        center = (self.sim.data.get_body_xpos("jaco_link_finger_1") + self.sim.data.get_body_xpos("jaco_link_finger_2") + self.sim.data.get_body_xpos("jaco_link_finger_3")) / 3
        centerd = np.linalg.norm(center - self.sim.data.get_body_xpos("target"))

        # for d in dist:
        #     reward -= d

        dist_sum = 0

        # easy
        reward -= centerd
        if centerd > 0.30:
            reward -= 100
            done = True

        if centerd < self.rewarding_distance:
            reward += (1/(centerd+1))
            #reward += 30
            #done = True

        # normal
        if dist[0] < 0.30 and dist[1] < 0.30 and dist[2] < 0.30:
            reward += 1

        # hard
        if dist[0] < 0.20 and dist[1] < 0.20 and dist[2] < 0.20:
            reward += 5

        # So hard
        if dist[0] < 0.15 and dist[1] < 0.15 and dist[2] < 0.15:
            if self.sim.data.get_body_xpos("target")[2] - 0.030989 > 0.05:
                reward += (self.sim.data.get_body_xpos("target")[2] - 0.030989) * 1000

        # collision
        #print(self.sim.data.active_contacts_efc_pos)
        # for collision in self.sim.data.active_contacts_efc_pos:
        #     if abs(collision) > 0.001:
        #         # collision!
        #         reward -= 50
        #         done = True
        #         print(collision)
        # if len(self.sim.data.active_contacts_efc_pos) > 7 and len(self.sim.data.active_contacts_efc_pos) < 17:
        #     done = True



        # if centerd == self.prev_centerd:
        #     reward += 10

        #
        # reward += (self.sim.data.get_body_xpos("target")[2] - 0.030989) * 100
        # if self.sim.data.get_body_xpos("target")[2] - 0.030989 > 0.05:
        #     reward += (self.sim.data.get_body_xpos("target")[2] - 0.030989) * 1000

        self.prev_centerd = centerd

        # Transform discrete actions to continuous controls
        # for i in range(self.num_actuators):
        #     '''
        #     0 = 0 velocity
        #     1 = small positive velocity
        #     2 = large positive velocity
        #     3 = small negative velocity
        #     4 = large negative velocity
        #     '''
        #     if a[i] == 0:
        #         new_control[i] = 0
        #     if a[i] == 1:
        #         new_control[i] = self.control_values[i] / 2
        #     if a[i] == 2:
        #         new_control[i] = self.control_values[i]
        #     if a[i] == 3:
        #         new_control[i] = -self.control_values[i] / 2
        #     elif a[i] == 4:
        #         new_control[i] = -self.control_values[i]

        # Do one step of simulation


        self.do_simulation(a)
        self.sum_reward += reward

        return self.get_obs(), reward, done
