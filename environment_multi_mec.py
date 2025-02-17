from .data_struct_multiagv import MEC, AGV
import numpy as np
from .critic import critic
import math
from .normalization import Normalization, RewardScaling

ack_mapping = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}

class MultiAgvOffloadingEnv:
    def __init__(self, mec_num=2, agv_num=16, num_channels=4, episode_limit=150, seed=None,
                 obs_entity_mode=False, state_entity_mode=False, map_name="", state_last_action=False,edge_only=False):
        # 信道数量、设备数量
        self.edge_only = edge_only
        self._seed = seed
        self.NUM_USERS = agv_num
        self.NUM_CHANNELS = num_channels
        self.NUM_MEC = mec_num
        self.episode_limit = episode_limit
        self.time_slot = 0
        self.obs_entity_mode = obs_entity_mode
        self.state_entity_mode = state_entity_mode
        self.mecs = []
        mec_radius = 50 
        mec_spacing = mec_radius * 2 
        for i in range(self.NUM_MEC):
            mec_x = (i * mec_spacing) + mec_radius
            mec_y = mec_radius
            self.mecs.append(MEC(mec_id=i, mec_x=mec_x, mec_y=mec_y))
        self.num_users_per_job = {'agv_num': self.NUM_USERS}
        user_counter = 0
        self.agents = []
        for job, num_users in self.num_users_per_job.items():
            for _ in range(num_users):
                if job == 'agv_num':
                    mec_index = np.random.randint(0, self.NUM_MEC) 
                    mec = self.mecs[mec_index]
                    agv_x, agv_y = generate_random_position_within_circle(mec.mec_x, mec.mec_y,
                                                                          mec.communication_range)
                    self.agents.append(AGV(user_id=user_counter, mec_index=mec_index, agv_x=agv_x, agv_y=agv_y))
                user_counter += 1
        # 画图
        # self.draw_mec_deployment()
        # 实验参数
        self.n_actions = self.NUM_CHANNELS + 1
        self.n_agents = self.NUM_USERS
        self.users_action = np.zeros([self.NUM_USERS], np.int32)
        self.users_observation = np.zeros([self.NUM_USERS], np.int32)
        # 网络环境参数
        self.computation_cycles = 31250  
        self.bandwidth = 5 * 1e6  
        self.noise_power = 1e-11  
        self.path_loss = 3  
        self.channel_gain = 5
        self.t_length = 5  
        self.last_ack = np.zeros([self.n_agents]) 
        self.last_action = np.zeros((self.n_agents, self.n_actions))  
        self.last_channel_utilization_state = np.zeros(self.NUM_CHANNELS)
        print("------use state normalization------")
        self.obs_norm = Normalization(shape=len(self.get_obs_agent(0)))

    def get_avail_agent_actions(self, agent_id):
        # 如果缓冲区不为空，则所有动作都可以选择
        if self.edge_only:
            if self.agents[agent_id].buffer:
                return [0] + [1] * (self.n_actions - 1)
            # 如果缓冲区为空，则只能选择不接入
            else:
                return [1] + [0] * (self.n_actions - 1)
        else:
            if self.agents[agent_id].buffer:
                return [1] * self.n_actions
            # 如果缓冲区为空，则只能选择不接入
            else:
                return [1] + [0] * (self.n_actions - 1)

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_critic_score(self):  # 获取每个智能体的critic指标
        agent_data_matrix = []
        for agent_id in range(self.n_agents):
            if self.agents[agent_id].buffer:
                prior_value = self.agents[agent_id].task_prior  # 修改为同一区域内的智能体数量
                delay_queue = self.agents[agent_id].buffer[0].delay_queue / self.agents[agent_id].latency_max
                buffer_length = len(self.agents[agent_id].buffer) / (self.agents[agent_id].latency_max/5 + 1)
            else:
                prior_value = 0
                buffer_length = 0
                delay_queue = 0
            # rand = 1e-6 * round(np.random.uniform(0.9, 1.1), 2)  # 与优先级相关的随机数，防止出现nan
            critic_value = np.array([prior_value + 1e-6 * round(np.random.uniform(0.9, 1.1), 2),
                                     delay_queue + 1e-6 * round(np.random.uniform(0.9, 1.1), 2),
                                     buffer_length + 1e-6 * round(np.random.uniform(0.9, 1.1), 2)])  # 每个智能体的critic指标
            agent_data_matrix.append(critic_value)
        agent_data_matrix = np.array(agent_data_matrix)  # 求出Critic矩阵
        self.score_list = critic(agent_data_matrix)  # 计算权重
        if math.isnan(self.score_list[1]):  # 判断是否有异常数据
            print(agent_data_matrix)
            print(self.score_list)

    def calculate_offload_delay(self, agent):

        channel_gain_linear = 10 ** (self.channel_gain / 10)
        user_mec_distance = np.sqrt((agent.agv_x - self.mecs[agent.mec_index].mec_x) ** 2 + (
                agent.agv_y - self.mecs[agent.mec_index].mec_y) ** 2) 
        path_loss_db = 128.1 + 37.6 * np.log10(user_mec_distance+0.1)
        path_loss_linear = self.path_loss ** (-path_loss_db / 10)

        transmit_rate = self.bandwidth * np.log2(
            1 + (channel_gain_linear * agent.transmit_power * path_loss_linear) / self.noise_power)
        transmit_delay = agent.buffer[0].data_size / transmit_rate * 1000
        compute_delay = (self.computation_cycles * agent.buffer[0].data_size / self.mecs[
            agent.mec_index].mec_compute_cap) * 1000  
        offload_delay = round(transmit_delay + compute_delay, 2)

        return offload_delay

    def get_agent_inf(self, agent_id):
        if self.agents[agent_id].buffer:
            job = self.agents[agent_id].buffer[0]  
            # prior_value = self.agents[agent_id].task_prior  
            data_delay = round((job.data_size * self.computation_cycles) / self.agents[agent_id].user_compute_cap * 1000) 
            computation_cycles = self.calculate_offload_delay(self.agents[agent_id]) 
            data_size = job.data_size
            # computation_cycles = self.computation_cycles
            remaining_delay = job.delay_threshold  
            buffer_length = len(self.agents[agent_id].buffer)
        else:
            # prior_value = 0
            data_delay = 0
            data_size = 0
            computation_cycles = 0
            remaining_delay = 0
            buffer_length = 0
        # critic_values = self.score_list[agent_id]
        # agent_inf = np.array(
        #     [self.agents[agent_id].mec_index, data_size, computation_cycles, remaining_delay,
        #      buffer_length])
        agent_inf = np.array(
            [data_size, data_delay, computation_cycles, remaining_delay,buffer_length])
        return agent_inf

    def get_obs_agent(self, agent_id):
        if self.obs_entity_mode:
            entity_obs = []
            for i in range(self.n_agents):
                if self.agents[agent_id].mec_index == self.agents[i].mec_index: 
                    obs_entity_inf = self.get_agent_inf(i)
                    is_self = 1 if i == agent_id else 0
                    is_self = np.array([is_self])
                    entity_obs_i = np.concatenate(
                        (
                            # self.last_action[agent_id],
                            # [self.last_ack[i]],
                            ack_mapping[self.last_ack[i]],
                            # self.last_channel_utilization,
                            obs_entity_inf,
                            is_self,
                        )
                    )
                    # entity_obs_i = [ack,data_size, computation_cycles, remaining_delay, buffer_length,is_self]
                else: 
                    entity_obs_i = np.zeros(len(self.get_agent_inf(0)) + 4)
                entity_obs.append(entity_obs_i)
            # print(entity_obs)
            return np.concatenate(entity_obs)
        else:
            obs_inf = self.get_agent_inf(agent_id)
            agent_obs = np.concatenate(
                (
                    # self.last_action[agent_id],
                    [self.last_ack[agent_id]],
                    # self.last_channel_utilization,
                    obs_inf,
                )
            )
            return agent_obs

    def get_obs(self):
        agents_obs = [self.obs_norm(self.get_obs_agent(i)) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        agent_states = np.stack([self.get_agent_inf(i) for i in range(self.n_agents)])
        # state = np.concatenate(self.get_obs(), axis=0).astype(
        #     np.float32
        # )
        ack_one_hot = np.array([ack_mapping[ack] for ack in self.last_ack])
        state = np.concatenate(
            (
                # self.last_action.flatten(),
                # self.last_ack,
                ack_one_hot.flatten(),
                # self.last_channel_utilization_state,
                agent_states.flatten()
            ),
        )
        # state = self.state_norm(state)  
        return state

    def reset_user(self):
        for agent in self.agents:
            mec_index = np.random.randint(0, self.NUM_MEC)  # 为每个AGV分配一个对应的MEC
            mec = self.mecs[mec_index]
            agent.agv_x, agent.agv_y = generate_random_position_within_circle(mec.mec_x, mec.mec_y,
                                                                              mec.communication_range)
            agent.buffer = []
            agent.success_job = 0
            agent.task_num = 0
            agent.task_success = 0
            agent.remain_delay = 0  
            agent.generate_job()

    def reset(self):
        # self.score_list = []
        self.reset_user()
        self.time_slot = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.last_ack = np.zeros([self.n_agents]) 
        # self.last_channel_utilization = np.zeros(self.NUM_CHANNELS)  
        # self.last_channel_utilization_state = np.zeros(self.NUM_CHANNELS)  
        return self.get_obs(), self.get_state()

    def get_reward(self):

        access_reward = 0
        for agent, ack in zip(self.agents, self.last_ack):
            if ack == 1:
                access_reward += 2
            elif ack == -1:
                access_reward += -1
            else:
                access_reward += 0

        delay_reward = 0

        overtime_penalty = 0
        for agent, ack in zip(self.agents, self.last_ack):
            if not agent.buffer:
                continue
            # print("计算奖励")
            local_delay = round((self.computation_cycles * agent.buffer[0].data_size / agent.user_compute_cap) * 1000,
                                2)  # m转ms
            if ack == 0:  
                if agent.buffer[0].delay_threshold - local_delay > 0:
                    agent.task_success += 1
                    agent.remain_delay += agent.latency_max - agent.buffer[0].delay_threshold + local_delay
                
                else:
                    overtime_penalty += agent.latency_max #(agent.buffer[0].delay_threshold - local_delay)

            elif ack == -1: 
                if agent.buffer[0].delay_threshold - self.t_length <= 0:
                    overtime_penalty += agent.latency_max  
            else: 
                access_reward += 1
                channel_gain_linear = 10 ** (self.channel_gain / 10)
                user_mec_distance = np.sqrt((agent.agv_x - self.mecs[agent.mec_index].mec_x) ** 2 + (
                        agent.agv_y - self.mecs[agent.mec_index].mec_y) ** 2) 
                path_loss_db = 128.1 + 37.6 * np.log10(user_mec_distance+0.1)
                path_loss_linear = self.path_loss ** (-path_loss_db / 10)

                transmit_rate = self.bandwidth * np.log2(
                    1 + (channel_gain_linear * agent.transmit_power * path_loss_linear) / self.noise_power)
                transmit_delay = agent.buffer[0].data_size / transmit_rate * 1000
                compute_delay = (self.computation_cycles * agent.buffer[0].data_size / self.mecs[
                    agent.mec_index].mec_compute_cap) * 1000  # MEC的计算时延
                offload_delay = round(transmit_delay + compute_delay, 2)
                delay_reward += local_delay - offload_delay

                if agent.buffer[0].delay_threshold - offload_delay > 0:
                    agent.task_success += 1
                    agent.remain_delay += agent.latency_max - agent.buffer[0].delay_threshold + offload_delay
                    # delay_reward += (agent.latency_max - agent.buffer[0].delay_threshold) + offload_delay
                    # overtime_penalty += agent.buffer[0].delay_threshold - offload_delay
                else:
                    overtime_penalty += agent.latency_max  # (agent.buffer[0].delay_threshold - offload_delay)
                # print(
                #     "{0}  overtime_penalty:{1:.2f}ms transmit_delay: {2:.2f} ms  compute_delay: {3:.2f} ms  "
                #     "delay_reward: {4:.2f}".format(agent.type, overtime_penalty, transmit_delay, compute_delay,
                #                                    delay_reward))

        reward = delay_reward - overtime_penalty
        # reward = access_reward # 测绘
        # print("access_reward", access_reward)
        # print("delay_reward", delay_reward)
        # print("overtime_penalty", overtime_penalty)
        return reward, delay_reward, overtime_penalty

    def update_users(self, agent, ack):
        mec_index = np.random.randint(0, self.NUM_MEC)  # 为每个AGV分配一个对应的MEC
        mec = self.mecs[mec_index]
        agent.agv_x, agent.agv_y = generate_random_position_within_circle(mec.mec_x, mec.mec_y,
                                                                          mec.communication_range)
        if ack[agent.user_id] != -1 and len(agent.buffer) > 0:
            agent.buffer.pop(0)
        buffer_copy = agent.buffer.copy() 
        for job in buffer_copy:
            job.delay_threshold -= 5  
            if job.delay_threshold <= 0:
                agent.buffer.remove(job)
        agent.generate_job()

    def step(self, actions):
        assert len(actions) == len(self.agents)

        info = {}
        conflict = 0
        terminated = False
        self.time_slot += 1
        actions_int = [int(a) for a in actions]
        actions_int = np.array(actions_int)
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        channel_alloc_frequency = [np.zeros(self.NUM_CHANNELS + 1) for _ in range(self.NUM_MEC)]
        # channel_alloc_frequency = np.bincount(actions_int, minlength=self.NUM_CHANNELS + 1)
        for mec_index in range(self.NUM_MEC):
            agv_in_range = [agent for agent in self.agents if agent.mec_index == mec_index]
            agv_actions = [actions_int[agent.user_id] for agent in agv_in_range]
            channel_alloc_frequency_local = np.bincount(agv_actions, minlength=self.NUM_CHANNELS + 1)
            channel_alloc_frequency_local[channel_alloc_frequency_local > 1] = 0
            channel_alloc_frequency[mec_index] += channel_alloc_frequency_local
        channel_utilization_rates = [sum(channel_alloc / self.NUM_CHANNELS) for channel_alloc in
                                         channel_alloc_frequency]
        channel_utilization_rate = sum(channel_utilization_rates) / self.NUM_MEC
        ack = np.zeros(self.n_agents, dtype=int)
        for agent in self.agents:
            action = actions_int[agent.user_id]
            mec_index = agent.mec_index
            if action == 0: 
                ack[agent.user_id] = 0
            elif channel_alloc_frequency[mec_index][action] == 1:  
                ack[agent.user_id] = 1
            else: 
                ack[agent.user_id] = -1
                conflict += 1
        self.last_ack = ack
        reward, info["delay_reward"], info["overtime_penalty"] = self.get_reward()
        info["reward"] = reward
        for agent in self.agents:
            # if agent.type == 'Normal':
            #     self.update_users(agent, ack)
            # elif agent.type == 'LowLatency':
            #     self.update_users(agent, ack)
            # else:
            self.update_users(agent, ack)

        info["channel_utilization_rate"] = channel_utilization_rate
        info["conflict_ratio"] = conflict / self.n_agents
        if self.time_slot == self.episode_limit:
            terminated = True
            info["episode_limit"] = True
            task_num_info = self.get_task_num()
            info["task_completion_rate"] = task_num_info["high_task_rate"]
            # info["high_task_rate"] = task_num_info["high_task_rate"]
            # info["mid_task_rate"] = task_num_info["mid_task_rate"]
            # info["low_task_rate"] = task_num_info["low_task_rate"]
            info["task_completion_delay"] = task_num_info["a_high_task_remain_delay"]
            # info["mid_task_remain_delay"] = task_num_info["a_mid_task_remain_delay"]
            # info["low_task_remain_delay"] = task_num_info["a_low_task_remain_delay"]
            # info["all_delay"] = task_num_info["all_delay"]
        return reward, terminated, info

    def get_task_num(self):
        high_task_num = 0
        # mid_task_num = 0
        # low_task_num = 0
        high_task_success_num = 0
        # mid_task_success_num = 0
        # low_task_success_num = 0
        high_task_remain_delay = 0
        # mid_task_remain_delay = 0
        # low_task_remain_delay = 0
        task_num_info = {}
        for agent in self.agents:
            #if agent.type == 'AGV':
                high_task_num += agent.task_num
                high_task_success_num += agent.task_success
                high_task_remain_delay += agent.remain_delay
            # elif agent.type == 'LowLatency':
            #     mid_task_num += agent.task_num
            #     mid_task_success_num += agent.task_success
            #     mid_task_remain_delay += agent.remain_delay
            # else:
            #     low_task_num += agent.task_num
            #     low_task_success_num += agent.task_success
            #     low_task_remain_delay += agent.remain_delay

        task_num_info["high_task_rate"] = high_task_success_num / high_task_num
        # task_num_info["mid_task_rate"] = mid_task_success_num / mid_task_num
        # task_num_info["low_task_rate"] = low_task_success_num / low_task_num
        # task_num_info["all_task_rate"] = (high_task_success_num + mid_task_success_num + low_task_success_num) / (
        #         high_task_num + mid_task_num + low_task_num)
        if high_task_success_num != 0:
            task_num_info["a_high_task_remain_delay"] = high_task_remain_delay / high_task_success_num
        else:
            task_num_info["a_high_task_remain_delay"] = 0
        # if mid_task_success_num != 0:
        #     task_num_info["a_mid_task_remain_delay"] = mid_task_remain_delay / mid_task_success_num
        # else:
        #     task_num_info["a_mid_task_remain_delay"] = 0
        # if low_task_success_num != 0:
        #     task_num_info["a_low_task_remain_delay"] = low_task_remain_delay / low_task_success_num
        # else:
        #     task_num_info["a_low_task_remain_delay"] = 0
        # if high_task_success_num != 0 or mid_task_success_num != 0 or low_task_success_num != 0:
        #     task_num_info["all_delay"] = (high_task_remain_delay + mid_task_remain_delay + low_task_remain_delay) / (
        #             high_task_success_num + mid_task_success_num + low_task_success_num)
        # else:
        #     task_num_info["all_delay"] = 0
        return task_num_info

    def seed(self):
        """Returns the random seed used by the environment."""
        return self._seed

    def get_env_info(self):
        env_info = {
            "state_shape": len(self.get_state()),
            # "obs_shape": len(self.get_obs()[0])*self.n_agents,
            "obs_shape": len(self.get_obs()[0]),
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "n_entities": self.n_agents,
        }
        if self.obs_entity_mode:
            env_info["obs_entity_feats"] = int(
                len(self.get_obs()[0]) / self.n_agents  # 矩阵的一行
            )
        if self.state_entity_mode:
            env_info["state_entity_feats"] = int(
                len(self.get_state()) / self.n_agents
            )
        return env_info

    def get_stats(self):
       pass

    def close(self):
        pass

    def draw_mec_deployment(self, filename="mec_deployment.png"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        for mec in self.mecs:
            circle = plt.Circle((mec.mec_x, mec.mec_y), mec.communication_range, color='r', fill=False)
            ax.add_artist(circle)
            ax.plot(mec.mec_x, mec.mec_y, 'rs', markersize=10)

        for idx, agent in enumerate(self.agents):
            ax.plot(agent.agv_x, agent.agv_y, 'bo', markersize=5)
            ax.text(agent.agv_x, agent.agv_y, f'AGV {idx}', fontsize=9, ha='center', va='bottom')

        ax.set_xlim(0, 100 * len(self.mecs))
        ax.set_ylim(0, 50 * len(self.mecs))

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('MEC Deployment')
        ax.set_aspect('equal')

        plt.show()
        plt.close(fig)  # 关闭图像以释放内存
        plt.savefig(filename, dpi=300, bbox_inches='tight')
