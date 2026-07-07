import ale_py  # Atari 환경 등록
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import ( # API 제공
    ResizeObservation, # (210, 160, 3) -> (84, 84, 1)
    GrayscaleObservation, # RGB -> Gray 
    FrameStackObservation, # state 4개 (frame) -> one state
)
from collections import deque
import random

# MODEL_PATH = "breakout_dqn.pth"

def make_env(render_mode="rgb_array"):
    # Atari Breakout 게임 생성 -> RGB to GRAY -> (84x84)변환 -> 4프레임 stack -> DQN입력 변환
    env = gym.make("ALE/Breakout-v5", render_mode=render_mode) # (210, 160, 3) uint8
    env = GrayscaleObservation(env) # (210, 160, 3) -> (210, 160, 1)
    env = ResizeObservation(env, (84, 84)) # (210, 160, 1) -> (84, 84, 1)
    env = FrameStackObservation(env, 4) # (84, 84) -> (4, 84, 84)
    return env

# DQN transition : (sate, action, reward, next_state, done)
# state : (4, 84, 84)
# action : 4 (0 = Nothing, 1 = Fire, 2 = Right, 3 = Left)
# reward : float (벽돌 o : +1.0, 벽돌 x : 0.0)
# next_state: action이후의 state (4, 84, 84)
# done: True/False (episode가 끝났는지?)
# Terminated (규칙상 종료) / Truncated (Timeout 등..)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        # Deque -> FIFO

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d)) # 하나의 tuple로 저장

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # buffer에서 batch_size만큼 랜덤 추출
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, action):
        self.num_actions = action
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_size = 100000
        self.learning_starts = 20000 # 학습 시작전 최소 buffer size
        self.target_update_period = 2000 # target Network updata pariod
        self.train_step = 0 # step counter -> 이걸로 target Network update 결정
        self.epsilon = 1.0 # random action 확률
        self.epsilon_min = 0.01 # 최솟값
        self.epsilon_decay_steps = 500000 # 1.0 -> 0.01 까지 줄어드는 step수
        self.memory = ReplayBuffer(self.buffer_size) # buffer

        self.agent_net = nn.Sequential( 
            # input (batch, 4, 84, 84), output (batch, num_actions)
            # output_size = floor((input_size - kernel_size) / stride) + 1
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # (batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (batch, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (batch, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), # (batch, 64, 7, 7) -> (batch, 64*7*7)
            nn.Linear(3136, 512), # (batch, 512)
            nn.ReLU(),
            nn.Linear(512, self.num_actions), # (batch, action_space:4)
        )
        self.target_net = nn.Sequential( 
            # input (batch, 4, 84, 84), output (batch, num_actions)
            # output_size = floor((input_size - kernel_size) / stride) + 1
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # (batch, 32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # (batch, 64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # (batch, 64, 7, 7)
            nn.ReLU(),
            nn.Flatten(), # (batch, 64, 7, 7) -> (batch, 64*7*7)
            nn.Linear(3136, 512), # (batch, 512)
            nn.ReLU(),
            nn.Linear(512, self.num_actions), # (batch, action_space:4)
        )
        self.target_net.load_state_dict(self.agent_net.state_dict())
        # target_net.parameters = agent_net.parameters
        self.target_net.eval() # target_net은 train안할것

        self.optimizer = optim.Adam(self.agent_net.parameters(), lr=1e-4)
        self.loss_fn = nn.MSELoss() 

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions) # random action
        # else
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # state: (4, 84, 84) numpy -> (1, 4, 84, 84) torch
        with torch.no_grad():
            # agent_net.eval()?
            q_values = self.agent_net(state_t) # forward 
        return int(torch.argmax(q_values, dim=1).item())
        # return Q_max

    def train_model(self):
        if len(self.memory) < self.learning_starts:
            return None
        # buffer에 최소치만큼(learning_starts) 채우고나서 train시작

        s, a, r, ns, d = self.memory.sample(self.batch_size) # buffer에서 batch만큼 sampling
        # return [five array] (numpy)
        s_t = torch.tensor(s, dtype=torch.float32) # (batch, 4, 84, 84)
        a_t = torch.tensor(a, dtype=torch.int64) # (batch,)
        r_t = torch.tensor(r, dtype=torch.float32) # (batch,)
        ns_t = torch.tensor(ns, dtype=torch.float32) # (batch, 4, 84, 84)
        d_t = torch.tensor(d, dtype=torch.float32) # (batch,)

        # Ballman Equation : Q = [R + gamma*V - Q]
        q_all = self.agent_net(s_t) # (32, 4)
        # q_all[0] = [0.3, 0.8, 2.1, 1.5] 4 action for 1 state
        # q_all[1] = [1.0, 0.2, 0.4, 3.0]
        # ... shape: (batch, 4) = Net return값
        # a_t = [1, 3, 2, ...] : sample0의 action = 1, sample1의 action = 3
        # ... shape: (batch,)
        # Goal q_sa = [0.8, 3.0, 0.1 ...] a_t를 보고 선택한 Q값 shape (batch,)
        q_sa = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1) # (32,)
        # a_t.unsqueeze(1) : a_t (batch,) -> (batch,1)
        # (batch, 4)랑 gather하기위해
        # gather(dim=1, index): q_all[0..batch]에서 a_t값의 index를 선택
        # shape: (batch, 1)
        # .squeeze(1): (batch, 1) -> (batch,)

        with torch.no_grad():
            next_q_all = self.target_net(ns_t) # Q(s')
            # next_q_all = [[0.3, 0.8, 2.1, 1.5], sample0
            #                [1.0, 0.2, 0.4, 0.3], sample1
            #                [...]
            #                ...]
            next_q_max = next_q_all.max(dim=1)[0] # maxQ(s') = V(s') 
            # .max(dim=1): action axis기준 max. ex) [0.3, 0.8, 2.1, 1.5]중 2.1
            # torch max return value = (values: 최댓값, indices: 최댓값 index) 
            target = r_t + self.gamma * (1.0 - d_t) * next_q_max
            # r + g*(1-d_t)*V(s') -> if d_t = 1 (terminated) then V(s') = 0

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1 # train count +1
        if self.train_step % self.target_update_period == 0: # update period
            self.target_net.load_state_dict(self.agent_net.state_dict())
            # copy target_net.parameters = agent_net.parameters

        return float(loss.item())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min: # epsilon최소값 아니라면 감소
            decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
            # decay = (최소까지 남은 양) / 남은 step수
            self.epsilon = max(self.epsilon_min, self.epsilon - decay)
            # epsilon -> epsilon_min ? 몇번을 굴려도 못가게 만드는게 원칙?

# debug
def debug_obs(obs):
    arr = np.array(obs)
    print(arr.shape, arr.dtype, arr.min(), arr.max())

def debug_action(action):
    if 0 < action and action < 4:
        print(f"Action: {action} (valid)")
    else:
        print(f"Action: {action} (invalid)")

def debug_buffer(memory, learning_starts):
    if len(memory) < learning_starts:
        print(f"Buffer size: {len(memory)} < {learning_starts} (learning_starts)")
    else:
        print(f"Buffer size: {len(memory)} >= {learning_starts} (learning_starts)")

def debug_train_step(loss, train_step):
    if loss is None:
        print("Train skipped: Not enough samples in buffer.")
    else:
        print(f"Train step: {train_step}, Loss: {loss}")

def preprocess_obs(obs):
    # FrameStackObservation 결과: (4, 84, 84), uint8():int (0~255 pixel)
    return np.array(obs, dtype=np.float32) / 255.0
    # object -> numpy array
    # dtype -> float32
    # 0~255값을 255.0으로 나누기 -> Normalize
    # Normalize?: 0~255 scale -> cause gradient/weight update 불안정

if __name__ == "__main__":
    env = make_env()
    agent = DQNAgent(action=env.action_space.n)
    # action_space type : Space
    # .n은 Discrete에서만 사용 (action수 countable)
    # type 추론 error로 인함

    for episode in range(50000):
        # game step = 0
        obs, info = env.reset() # 환경 reset
        state = preprocess_obs(obs) # numpy array변환
        done = False # terminate false
        episode_reward = 0.0 # total commulative reward 0

        while not done:
            action = agent.get_action(state) # state에 따라 action 선택
            # AgentNet 사용 + stocastic policy

            next_obs, reward, terminated, truncated, info = env.step(action)
            # 환경에 action전달 -> 이후 얻어지는 (state, reward, ...)
            done = terminated or truncated # 사망/클리어 -> done = True
            next_state = preprocess_obs(next_obs) # next_state 또한 numpy array변환

            agent.memory.push(state, action, reward, next_state, done)
            # buffer에 push
            agent.train_model() # buffer 최소치 만족 -> batch만큼 sampling후 튜닝
            agent.decay_epsilon() # 1 step마다 epsilon값 하향

            state = next_state # state이동
            episode_reward += float(reward)
            # game_step += 1
        if episode % 100 == 0:
            print(f"ep={episode} reward={episode_reward:.1f} eps={agent.epsilon:.3f}")

    # torch.save(agent.agent_net.state_dict(), MODEL_PATH)
    # agent_net.parameter MODEL_PATH에 저장
    # print(f"Saved model to {MODEL_PATH}")

    env.close()

# Naive DQN : target_net 안쓰는 버전

# Double DQN : overestimate 줄이기위함?
# next_q_all = target_net(ns_t) -> next_action = agent_net(ns_t).argmax(dim=1)
# next_q_max = next_q_all.max(dim=1)[0] -> next_q = target_net(ns_t).gather(1, next_actions)

# Dueling DQN : Q -> V + A
# CNN -> Flatten 이후 Value stream(batch, 1) / Advantage stream 나눔(batch, 4)
# Q(s,a) = V(s) + (A(s,a) - mean(A(s,;))
# mean(A(s,;)) 즉 baseline을 빼주면서 Advantage mean을 0 근처로 scale

# Prioritized Experience Replay : Buffer Sampling에 가중치 추가
# batch = random.sample(buffer,32) -> priority 추가

# n-step DQN : TD 형식의 1-step reward 대신 n-step return 사용
# target = r + gamma*(1-d)*max Q_target(s') -> r(t) + gamma*r(t+1) + gamma**2*r(t+2) ... + gamma**n max Q_target(s(t+n))

# Distributional DQN / Noisy Networks / Rainbow DQN / Deep Recurrent DQN / Continuous DQN