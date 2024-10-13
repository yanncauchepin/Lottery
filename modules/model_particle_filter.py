import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

root_path = "/home/yanncauchepin/Git/Lottery"

class ParticleFilter():
    def __init__(self, num_particles, numbers):
        self.num_particles = num_particles
        self.numbers = numbers
        self.particles = np.random.dirichlet(np.ones(numbers), num_particles) 
        self.weights = np.ones(num_particles) / num_particles 

    def predict(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)  # Reset weights

    def update(self, z):
        # Update weights based on the likelihood of the observation
        for i in range(self.num_particles):
            self.weights[i] *= self.likelihood(z, self.particles[i])

        # Normalize the weights
        self.weights += 1.e-300  # Prevent division by zero
        self.weights /= np.sum(self.weights)

        # Resample particles based on updated weights
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights after resampling

    def likelihood(self, z, particle):
        # Likelihood function (could be Gaussian or any other distribution)
        # For simplicity, we will use a softmax likelihood here
        return np.exp(np.sum(z * np.log(particle + 1e-10)))  # Small value to prevent log(0)

    def get_estimate(self):
        # Return the mean of the particles
        return np.mean(self.particles, axis=0)

def binary_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return cross_entropy

def categorical_crossentropy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, 1e-12, 1.0)
    cross_entropy = -np.sum(y_true * np.log(y_pred))
    return cross_entropy

def meta_modeling(lottery, df, num_particles, numbers):
    files = glob.glob(os.path.join(root_path, f'history_particle_filter/{lottery}/*'))
    for file in files:
        if os.path.exists(file):
            os.remove(file)

    particle_filter = ParticleFilter(num_particles, numbers)
    i = 0
    for date, draw in df.iloc[-50:].iterrows():
        i += 1
        particle_filter.predict()
        estimate = particle_filter.get_estimate()
        print(f'{date}: {binary_crossentropy(draw, estimate)}')

        plt.figure()
        plt.bar(range(len(draw)), draw * 2 * np.max(estimate))
        plt.bar(range(len(estimate)), estimate)
        os.makedirs(os.path.join(root_path, f'history_particle_filter/{lottery}'), exist_ok=True)
        plt.savefig(os.path.join(root_path, f'history_particle_filter/{lottery}/{date}.png'))
        plt.close()
        particle_filter.update(np.array(draw))
        
    estimate = particle_filter.get_estimate()
    proba = pd.DataFrame(estimate, index=range(1, numbers + 1))
    proba = proba.sort_values(by=0, ascending=False)
    return proba

if __name__ == '__main__':
    df = pd.read_csv('data/all_concat_one_hot_ball_loto.csv', index_col=0)
    result = meta_modeling("loto_ball", df, 100000, 49)
    print(result)