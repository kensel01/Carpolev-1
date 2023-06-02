import gym
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam 

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
#Creo un entorno de la biblioteca Gym, especificamente "CartPole-v1"
env = gym.make("CartPole-v1")

#Se obtienen las dimensiones del espacio de observacion del entorno y el numero de acciones posibles.
states= env.observation_space.shape[0]
actions = env.action_space.n 
#Se crea un modelo de red neuronal utilizando la biblioteca keras, esta tiene una capa de entrada Flatten recibe el estado del entorno.
#dos capas ocultas dense con 24 unidades y funcion de activacion relu.
#y por ultimo una capa de salida con tantas unidades como acciones posibles en el entorno y funcion de activacion lineal
model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))
#se crea un agente DQN(deep Q-network) el agente utiliza el modelo de red neuronal definido anteriormente.
agent = DQNAgent (
    model= model,
    memory= SequentialMemory(limit=50000, window_length=1),
    policy= BoltzmannQPolicy(),
    nb_actions= actions,
    nb_steps_warmup=10,
    target_model_update = 0.01
)
#se compila el agente utilizando el optiimizador adam con una tasa de aprendizaje de 0.001 y se especifica que la metrica a seguir durante el entrenamiento es el error absoluto medio(mae)
agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
#se entrena al agente utilizando el entrno env durante 100000 pasos, la opcion visualize false desabilita la ventana durante el entrenamiento
#verbose 1 es para mostrar la infromacion detallada durante el entrenamiento
agent.fit(env, nb_steps=100000, visualize= False, verbose =1)
#Se evalúa el agente en el entorno durante 10 episodios utilizando el método test. La opción visualize se establece en True para visualizar el comportamiento del agente durante la evaluación.
results= agent.test(env, nb_episodes=10, visualize= True)
#imprimen el promedio de las recompensas obtenidas durante la evaluacion del agente
print(np.mean(results.history["episode_reward"]))

env.close()