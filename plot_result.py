import matplotlib.pyplot as plt
import numpy as np

f1 = open("reward.txt", 'r')

dppo_list = []
def read_reward(reward_list):
	while True:
		line = f1.readline()
		if not line: break
		reward_list.append(float(line))

read_reward(dppo_list)
x= np.arange(len(dppo_list))
y=dppo_list

plt.plot(x,y,'b',label='dppo')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DPPO')
plt.legend(loc='lower right')
plt.show()

f1.close()
