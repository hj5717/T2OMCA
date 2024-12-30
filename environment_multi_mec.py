from .data_struct_multiagv import MEC, AGV
import numpy as np
from .critic import critic
import math
from .normalization import Normalization, RewardScaling

ack_mapping = {-1: [1, 0, 0], 0: [0, 1, 0], 1: [0, 0, 1]}


def generate_random_position_within_circle(center_x, center_y, radius):
    # 生成一个位于圆内的随机位置
    angle = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(0, radius)
    x = round(center_x + distance * np.cos(angle), 1)
    y = round(center_y + distance * np.sin(angle), 1)
    return x, y


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
        # 图结果的MDP参数
        self.obs_entity_mode = obs_entity_mode
        self.state_entity_mode = state_entity_mode
        # 生成MEC服务器
        # MEC服务器的数量为（mec_num）个，每个MEC有5个可通信信道，覆盖范围为50m，且覆盖范围不重叠，设定MEC的位置为x,y.
        self.mecs = []
        mec_radius = 50  # MEC的通信范围为50米
        mec_spacing = mec_radius * 2  # 设置相邻MEC之间的距离为通信范围的两倍
        for i in range(self.NUM_MEC):
            mec_x = (i * mec_spacing) + mec_radius
            mec_y = mec_radius
            self.mecs.append(MEC(mec_id=i, mec_x=mec_x, mec_y=mec_y))
        # 生成智能体（agv）
        self.num_users_per_job = {'agv_num': self.NUM_USERS}
        user_counter = 0
        self.agents = []
        # 每个AGV的初始位置在对应的MEC服务器范围内
        for job, num_users in self.num_users_per_job.items():
            for _ in range(num_users):
                if job == 'agv_num':
                    mec_index = np.random.randint(0, self.NUM_MEC)  # 为每个AGV分配一个对应的MEC
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
        self.computation_cycles = 31250  # 所需计算资源 (MHz/Byte * Bytes) 0.25MHz/Byte 转换为cycles/bit
        self.bandwidth = 5 * 1e6  # 通信带宽,单位Hz
        self.noise_power = 1e-11  # 噪声功率,单位W
        self.path_loss = 3  # 路径损耗
        self.channel_gain = 5
        self.t_length = 5  # 每个时隙长度定义为5ms
        self.last_ack = np.zeros([self.n_agents])  # 执行动作后各设备收到的ACK
        self.last_action = np.zeros((self.n_agents, self.n_actions))  # 上时隙动作
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
        """
        计算将任务卸载到MEC上的时延

        Args:
        agent (Agent): 代理对象

        Returns:
        float: 卸载时延(ms)
        """

        # 转换
        channel_gain_linear = 10 ** (self.channel_gain / 10)
        user_mec_distance = np.sqrt((agent.agv_x - self.mecs[agent.mec_index].mec_x) ** 2 + (
                agent.agv_y - self.mecs[agent.mec_index].mec_y) ** 2)  # agent_id是当前智能体，i是所以智能体在遍历
        # 计算路径损耗
        path_loss_db = 128.1 + 37.6 * np.log10(user_mec_distance+0.1)
        path_loss_linear = self.path_loss ** (-path_loss_db / 10)

        # 计算卸载到MEC的时延
        transmit_rate = self.bandwidth * np.log2(
            1 + (channel_gain_linear * agent.transmit_power * path_loss_linear) / self.noise_power)
        transmit_delay = agent.buffer[0].data_size / transmit_rate * 1000
        compute_delay = (self.computation_cycles * agent.buffer[0].data_size / self.mecs[
            agent.mec_index].mec_compute_cap) * 1000  # MEC的计算时延
        offload_delay = round(transmit_delay + compute_delay, 2)

        return offload_delay

    def get_agent_inf(self, agent_id):
        # 缓冲区队首的任务优先级、任务大小、所需计算资源
        if self.agents[agent_id].buffer:
            job = self.agents[agent_id].buffer[0]  # 获取缓冲区队首的任务
            # prior_value = self.agents[agent_id].task_prior  # 设备优先级
            data_delay = round((job.data_size * self.computation_cycles) / self.agents[agent_id].user_compute_cap * 1000)  # 任务大小 改成本地计算时延
            computation_cycles = self.calculate_offload_delay(self.agents[agent_id])  # 改成卸载到MEC的计算
            # 传输信道增益
            data_size = job.data_size
            # computation_cycles = self.computation_cycles
            remaining_delay = job.delay_threshold  # 缓冲区任务的最短剩余时延
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
            # 一个entity的obs
            entity_obs = []
            for i in range(self.n_agents):
                # 智能体的obs只包含同边缘服务器覆盖范围内的其它智能体，在边缘服务器外的智能体观测被设置为0
                if self.agents[agent_id].mec_index == self.agents[i].mec_index:  # 判断智能体的位置是否在同一个MEC服务器覆盖区域内
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
                else:  # 与该智能体不在同一个MEC服务器覆盖范围下
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
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        # agent_obs = self.obs_norm(agent_obs)  # 观测值归一化
        agents_obs = [self.obs_norm(self.get_obs_agent(i)) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self):
        # 历史信息：（last_action +  信道使用情况——全局 + ACK） + 所有设备信息：（优先级（1）+剩余时延（1）+ 平均剩余时延 + 缓冲区数据量（1））
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
        # state = self.state_norm(state)  # 观测值归一化
        return state

    def reset_user(self):
        # 清空所有设备的数据缓存
        for agent in self.agents:
            # 位置更新
            mec_index = np.random.randint(0, self.NUM_MEC)  # 为每个AGV分配一个对应的MEC
            mec = self.mecs[mec_index]
            agent.agv_x, agent.agv_y = generate_random_position_within_circle(mec.mec_x, mec.mec_y,
                                                                              mec.communication_range)
            agent.buffer = []
            agent.success_job = 0
            agent.task_num = 0
            agent.task_success = 0
            agent.remain_delay = 0  # 好像没啥用了,现在改为所完成任务的总时延
            # 产生新初始数据任务
            agent.generate_job()
        # 计算各设备指标
        # self.get_critic_score()

    def reset(self):
        # self.score_list = []
        self.reset_user()
        self.time_slot = 0
        self.last_action = np.zeros((self.n_agents, self.n_actions))
        self.last_ack = np.zeros([self.n_agents])  # 执行动作后各设备收到的ACK
        # self.last_channel_utilization = np.zeros(self.NUM_CHANNELS)  # 执行动作后信道利用情况
        # self.last_channel_utilization_state = np.zeros(self.NUM_CHANNELS)  # 全局角度观测到的信道利用情况，包含了每个信道被选择的次数
        return self.get_obs(), self.get_state()

    def get_reward(self):

        # 计算接入奖励 接入成功奖励+1，选择放弃接入奖励为0，接入失败奖励-1
        access_reward = 0
        for agent, ack in zip(self.agents, self.last_ack):
            if ack == 1:
                access_reward += 2
            elif ack == -1:
                access_reward += -1
            else:
                access_reward += 0

        # 如果设备选择了本地卸载，延迟奖励为0
        # 当任务卸载到MEC服务器，延迟奖励 = 本地计算时延 - （任务卸载到服务器的传输时延 + 任务在服务器的计算时延）
        # 计算各个设备选择本地计算时的计算时延（注意单位，时延单位为ms）
        # 计算各个设备选择卸载到MEC计算时的计算时延与传输时延
        # 延迟奖励
        delay_reward = 0
        # 超时惩罚
        overtime_penalty = 0
        for agent, ack in zip(self.agents, self.last_ack):
            # 判断缓冲区是否为空
            if not agent.buffer:
                # 对于缓冲区为空的情况,不需要计算任何延迟奖励或惩罚
                continue
            # 计算本地计算时延
            # print("计算奖励")
            local_delay = round((self.computation_cycles * agent.buffer[0].data_size / agent.user_compute_cap) * 1000,
                                2)  # m转ms
            if ack == 0:  # 本地执行,延迟奖励为0，计算惩罚(任务的剩余时延-所需要的时延)，小于时延大于deadline, agent.task_success+=1
                if agent.buffer[0].delay_threshold - local_delay > 0:
                    agent.task_success += 1
                    agent.remain_delay += agent.latency_max - agent.buffer[0].delay_threshold + local_delay
                    # overtime_penalty += agent.buffer[0].delay_threshold - local_delay
                    # delay_reward += agent.latency_max - agent.buffer[0].delay_threshold + local_delay
                else:
                    overtime_penalty += agent.latency_max #(agent.buffer[0].delay_threshold - local_delay)

            elif ack == -1:  # 没执行，后果很严重，直接计算惩罚
                if agent.buffer[0].delay_threshold - self.t_length <= 0:
                    overtime_penalty += agent.latency_max  # 没传上，需要再排队一下
            else:  # 卸载到MEC
                # 转换
                access_reward += 1
                channel_gain_linear = 10 ** (self.channel_gain / 10)
                user_mec_distance = np.sqrt((agent.agv_x - self.mecs[agent.mec_index].mec_x) ** 2 + (
                        agent.agv_y - self.mecs[agent.mec_index].mec_y) ** 2)  # agent_id是当前智能体，i是所以智能体在遍历
                # 计算路径损耗
                path_loss_db = 128.1 + 37.6 * np.log10(user_mec_distance+0.1)
                path_loss_linear = self.path_loss ** (-path_loss_db / 10)

                # 计算卸载到MEC的时延
                transmit_rate = self.bandwidth * np.log2(
                    1 + (channel_gain_linear * agent.transmit_power * path_loss_linear) / self.noise_power)
                transmit_delay = agent.buffer[0].data_size / transmit_rate * 1000
                compute_delay = (self.computation_cycles * agent.buffer[0].data_size / self.mecs[
                    agent.mec_index].mec_compute_cap) * 1000  # MEC的计算时延
                offload_delay = round(transmit_delay + compute_delay, 2)
                # 延迟奖励 = 本地计算时延 - 卸载到MEC的时延
                delay_reward += local_delay - offload_delay
                # 超时惩罚 时延大于deadline，则有惩罚，小于时延大于deadline, agent.task_success+=1

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
        # 位置更新
        mec_index = np.random.randint(0, self.NUM_MEC)  # 为每个AGV分配一个对应的MEC
        mec = self.mecs[mec_index]
        agent.agv_x, agent.agv_y = generate_random_position_within_circle(mec.mec_x, mec.mec_y,
                                                                          mec.communication_range)
        # 传输成功的数据任务出队
        if ack[agent.user_id] != -1 and len(agent.buffer) > 0:
            # 将成功传输的数据任务的剩余时延以及他们的优先级记录，并用某种方式在实验后展现出来
            # success_job_data = agent.buffer[0]  # 获取成功传输的任务数据
            # self.success_jobs_info.append((success_job_data.delay_queue, agent.task_prior))
            # agent.remain_delay += (agent.latency_max - agent.buffer[0].delay_queue)
            agent.buffer.pop(0)
        buffer_copy = agent.buffer.copy()  # create a copy of the buffer to avoid modifying it while iterating over it
        for job in buffer_copy:
            job.delay_threshold -= 5  # 一个时隙设置为5ms吧
            if job.delay_threshold <= 0:
                agent.buffer.remove(job)
        # 产生新的数据任务
        agent.generate_job()

    def step(self, actions):
        assert len(actions) == len(self.agents)
        # 初始化
        info = {}
        conflict = 0
        terminated = False
        self.time_slot += 1
        actions_int = [int(a) for a in actions]
        actions_int = np.array(actions_int)
        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]
        # 设备与信道的接入对应关系
        # 每个MEC服务器拥有自己的一组信道
        channel_alloc_frequency = [np.zeros(self.NUM_CHANNELS + 1) for _ in range(self.NUM_MEC)]
        # channel_alloc_frequency = np.bincount(actions_int, minlength=self.NUM_CHANNELS + 1)
        # 对于每个MEC服务器
        for mec_index in range(self.NUM_MEC):
            # 获取该MEC服务器覆盖范围内的AGV
            agv_in_range = [agent for agent in self.agents if agent.mec_index == mec_index]
            # 获取这些AGV的动作
            agv_actions = [actions_int[agent.user_id] for agent in agv_in_range]
            # 在该MEC服务器范围内进行信道竞争
            channel_alloc_frequency_local = np.bincount(agv_actions, minlength=self.NUM_CHANNELS + 1)
            # 部分观测信道状态,被多次选择的信道发生接入冲突,置零
            channel_alloc_frequency_local[channel_alloc_frequency_local > 1] = 0
            # 更新该MEC服务器的信道分配情况
            channel_alloc_frequency[mec_index] += channel_alloc_frequency_local
        # 信道利用率
        channel_utilization_rates = [sum(channel_alloc / self.NUM_CHANNELS) for channel_alloc in
                                         channel_alloc_frequency]
        channel_utilization_rate = sum(channel_utilization_rates) / self.NUM_MEC
        # 获取各设备接入情况
        # 如果设备的动作为0，那么选择了本地卸载，ack=0
        # 如果设备选择了卸载到边缘服务器，且发生了接入冲突，那么ack=-1
        # 如果设备选择了卸载到边缘服务器，并成功完成了信道接入，那么ack=1
        # 获取各设备接入情况
        ack = np.zeros(self.n_agents, dtype=int)
        for agent in self.agents:
            action = actions_int[agent.user_id]
            mec_index = agent.mec_index
            if action == 0:  # 本地卸载
                ack[agent.user_id] = 0
            elif channel_alloc_frequency[mec_index][action] == 1:  # 成功卸载到边缘服务器
                ack[agent.user_id] = 1
            else:  # 发生接入冲突
                ack[agent.user_id] = -1
                conflict += 1
        self.last_ack = ack
        # print("ack=",ack)
        reward, info["delay_reward"], info["overtime_penalty"] = self.get_reward()
        info["reward"] = reward
        # 状态更新，在缓冲区中超时的任务会被移除
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

        # 绘制 MEC 服务器和通信范围
        for mec in self.mecs:
            circle = plt.Circle((mec.mec_x, mec.mec_y), mec.communication_range, color='r', fill=False)
            ax.add_artist(circle)
            ax.plot(mec.mec_x, mec.mec_y, 'rs', markersize=10)

        # 绘制 AGV 智能体，并显示其ID
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

# # 环境测试
# from data_struct_multiagv import MEC, AGV
# import numpy as np
# from critic import critic
# import math
# from normalization import Normalization, RewardScaling
# env = MultiAgvOffloadingEnv(mec_num=2, agv_num=4, num_channels=1, episode_limit=150, seed=None,
#                  obs_entity_mode=False, state_entity_mode=False, map_name="", state_last_action=False)
# actions = [1,0,1,0]
# env.reset()
# print(env.step(actions))