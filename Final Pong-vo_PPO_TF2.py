import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import cv2

import threading
from threading import Thread, Lock
import time
import matplotlib

matplotlib.use('Agg')

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


def OurModel(input_shape, action_space, lr):
    X_input = Input(shape=input_shape)
    X = Flatten()(X_input)
    X = Dense(512, activation='elu', kernel_initializer='he_uniform')(X)
    action = Dense(action_space, activation='softmax', kernel_initializer='he_uniform')(X)
    value = Dense(1, activation='linear', kernel_initializer='he_uniform')(X)
    eps_clip = 0.2
    ENTROPY_LOSS = 0.01

    def ppo_loss(y_true, y_pred):
        A_i_gae, prediction, actions, td_target, values = y_true[:, :1], y_true[:, 1:1+action_space], y_true[:, 1+action_space:-2] , y_true[:,-2:-1],y_true[:,-1]
        prob = y_pred * actions
        old_prob = prediction * actions
        r = (prob + 1e-10) / (old_prob + 1e-10)
        c1 = r * A_i_gae
        c2 = K.clip(r, min_value=1 - eps_clip, max_value=1 + eps_clip) * A_i_gae
        loss = -K.mean(K.minimum(c1, c2) + ENTROPY_LOSS * -(prob * K.log(prob + 1e-10)))  # 수정된 부분

        return loss

    Actor = Model(inputs=X_input, outputs=action)
    Actor.compile(loss=ppo_loss, optimizer= RMSprop(lr=lr))

    Critic = Model(inputs=X_input, outputs=value)
    Critic.compile(loss='mse', optimizer= RMSprop(lr=lr))

    return Actor, Critic


class PPOAgent:
    def __init__(self, env_name):
        # Initialization
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.EPISODES, self.episode, self.max_average = 10000, 0, -21.0  # specific for pong
        self.lock = Lock()  # lock all to update parameters without other thread interruption
        self.lr = 0.0001

        self.ROWS = 80
        self.COLS = 80
        self.REM_STEP = 4
        self.EPOCHS = 10

        self.scores, self.episodes, self.average = [], [], []

        self.Save_Path = 'Models'
        self.state_size = (self.REM_STEP, self.ROWS, self.COLS)

        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.path = '{}_APPO_{}'.format(self.env_name, self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)

        # Actor, Critic 생성
        self.Actor, self.Critic = OurModel(input_shape=self.state_size, action_space=self.action_size, lr=self.lr)

    def v(self, x):
        return self.Critic.predict(x)

    def Calculate_Study(self, states, next_states, actions, rewards, predictions, dones):
        gamma = 0.99
        lmbda = 0.95

        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        done_mask_list = []
        for done in dones:
            done = 0 if done else 1
            done_mask_list.append([done])

        # TD_target, A_i_gae 계산
        pre_advantage = 0
        pre_advantages = np.zeros_like(rewards)
        td_target = rewards + gamma * self.v(next_states) * done_mask_list
        values = self.v(states)
        advantages = td_target - values
        
        

        for i in reversed(range(0, len(advantages))):
            pre_advantage = gamma * lmbda * pre_advantage + advantages[i][0]
            pre_advantages[i] = pre_advantage

        
        pre_advantages -= np.mean(pre_advantages)  # A_i_gae^2으로 Critic 업데이트
        pre_advantages /= np.std(pre_advantages)

        A_i_gae = np.vstack(pre_advantages)
        

        # 넘파이 배열에 전부 저장
        y_true = np.hstack([A_i_gae, predictions, actions, td_target, values])

        # Actor and Critic networks 학습
        self.Actor.fit(states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))
        self.Critic.fit(states, td_target, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(rewards))

    def act(self, state):  # NN에 s를 집어넣어서 나온 액션 확률에서 액션 샘플링
        prediction = self.Actor.predict(state)[0]  # Actor 모델을 이용해 주어진 상태에 대한 액션 확률 예측
        action = np.random.choice(self.action_size, p=prediction)  # 예측된 확률에 따라 액션 선택
        return action, prediction

    def imshow(self, image, rem_step=0):
        cv2.imshow("cartpole" + str(rem_step), image[rem_step, ...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self, frame, image_memory):
        # batchsize가 1일 때 Self.Rem_step개의 frame을 한 번에 처리
        if image_memory.shape == (1, *self.state_size):  # image_memory.shape가 (1,self.Rem__step, 80, 80)라면 -> (self.Rem_step, 80, 80)
            image_memory = np.squeeze(image_memory)  # 차원 줄이기

        frame_cropped = frame[35:195:2, ::2, :]  # 프레임의 크기를 (160, 120, 3)에서 (80, 80, 3)으로 축소
        if frame_cropped.shape[0] != self.COLS or frame_cropped.shape[1] != self.ROWS:
            frame_cropped = cv2.resize(frame, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)

        # converting to RGB (numpy way) 그레이스케일 이미지로 변환
        frame_rgb = 0.299 * frame_cropped[:, :, 0] + 0.587 * frame_cropped[:, :, 1] + 0.114 * frame_cropped[:, :, 2]

        # 이미지 흑백으로 변경해서 더 빨리 학습
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255

        # 0~1 사이 값으로 변환
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0

        # 데이터를 한 프레임씩 밀어내는 (dequeue) 작업을 수행
        image_memory = np.roll(image_memory, 1, axis=0)

        # 새로운 프레임을 저장, 메모리에서 유지
        image_memory[0, :, :] = new_frame

        return np.expand_dims(image_memory, axis=0)  # (배치 크기, 채널 수, 높이, 너비) 형태의 입력을 요구

    def step(self, action, env, image_memory):
        next_state, reward, done, info = env.step(action)
        next_state = self.GetImage(next_state, image_memory)
        return next_state, reward, done, info

    def reset(self, env):
        image_memory = np.zeros(self.state_size)
        frame = env.reset()
        for i in range(self.REM_STEP):
            state = self.GetImage(frame, image_memory)
        return state

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":  # much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path + ".png")
            except OSError:
                pass

        return self.average[-1]

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')

    def train(self, n_threads):
        self.env.close()
        # 한 쓰레드당 한 환경 Instantiate
        envs = [gym.make(self.env_name) for i in range(n_threads)]

        # 쓰레드 생성
        threads = [threading.Thread(
            target=self.train_threading,
            daemon=True,
            args=(self,
                  envs[i],
                  i)) for i in range(n_threads)]

        for t in threads:
            time.sleep(2)
            t.start()

        for t in threads:
            time.sleep(10)
            t.join()

    def train_threading(self, agent, env, thread):
        while self.episode < self.EPISODES:
            # 에피소드 reset
            score, done, SAVING = 0, False, ''
            state = self.reset(env)
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            while not done:
                action, prediction = agent.act(state)
                next_state, reward, done, _ = self.step(action, env, state)
                states.append(state)
                next_states.append(next_state)
                action_onehot = np.zeros([self.action_size])
                action_onehot[action] = 1
                actions.append(action_onehot)
                rewards.append([reward])
                predictions.append(prediction)
                dones.append(done)

                score += reward
                state = next_state

            self.lock.acquire()
            self.Calculate_Study(states, next_states, actions, rewards, predictions, dones)
            self.lock.release()

            # Update episode count
            with self.lock:
                average = self.PlotModel(score, self.episode)
                # 최고 모델만 saving
                if average >= self.max_average:
                    self.max_average = average
                    self.save()
                    SAVING = "SAVING"
                else:
                    SAVING = ""
                print("episode: {}/{}, thread: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES,
                                                                                          thread, score, average, SAVING))
                if self.episode < self.EPISODES:
                    self.episode += 1
        env.close()

    def test(self, Actor_name, Critic_name):
        self.load(Actor_name, Critic_name)
        for e in range(100):
            state = self.reset(self.env)
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state))
                state, reward, done, _ = self.step(action, self.env, state)
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, score))
                    break
        self.env.close()

    

if __name__ == "__main__":
    env_name = 'PongDeterministic-v4'
    # env_name = 'Pong-v0'
    agent = PPOAgent(env_name)
    #agent.run() # use as PPOagent.train(n_threads=6)
    # agent.train(n_threads=5) # use as APPO
    # agent.test('Models/Pong-v0_APPO_0.0001_Actor.h5', '')
    # agent.test('Models/PongDeterministic-v4_APPO_0.0001_Actor.h5', '')
