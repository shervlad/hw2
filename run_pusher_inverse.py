from push_env import PushingEnv


env = PushingEnv(ifRender=True)
distances = []
for i in range(10):
    dist = env.plan_inverse_model()
    distances.append(round(dist,4))
    env.reset()
print(distances)
val = input("press a key to finish")