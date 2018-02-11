a_list=[1., 2., 3.]
for a in a_list:
	with open("reward.txt", "a") as f:
		f.write(str(a) + '\n')
a_list = [4.,5.,6.]
for a in a_list:
	with open("reward.txt", "a") as f:
		f.write(str(a) + '\n')

