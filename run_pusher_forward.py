from push_env import PushingEnv
import sys

env = PushingEnv(ifRender=True,model='forward')

distances = []
for i in range(10):
    dist = env.plan_inverse_model()
    distances.append(round(dist,4))
    env.reset()
print(distances)
val = input("press a key to finish")