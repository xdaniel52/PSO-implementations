import numpy as np

class Particle:
     def __init__(self, position, value, velocity):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value


class PSO:
    def __init__(self, num_var, pop_size, c1, c2, epochs, range_of_params):
        self.num_var = num_var
        self.pop_size = pop_size
        self.w = 0.5 + np.random.random()/2
        self.c1 = c1
        self.c2 = c2
        self.epochs = epochs
        self.range_of_params = range_of_params
        self.global_best_position = 0
        self.global_best_value = 0
        self.particles = []
        
    def Init_particles(self):
        self.particles = []
        for i in range(pop_size):
            position = []
            velocity = []
            for dim in range(num_var):
                position.append(np.random.random()*(range_of_params[dim][1]-range_of_params[dim][0]) + range_of_params[dim][0]) 
                velocity.append(0.1 * position[dim]) 
            value = self.function(position)
            self.particles.append(Particle(position, value, velocity))
    
    def Start(self, function):
        self.function = function
        self.Init_particles()
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_value = self.particles[0].value.copy()
        for i in range(pop_size):
            if(self.particles[i].value < self.global_best_value):
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_value = self.particles[i].value.copy() 
               
        for epoch in range(epochs):
            self.Update_inertia_weight()
            self.Update_particles_velicity()
            self.Update_particles_position()
                       
        self.Print_result()
    
    def Print_result(self):
        print("best pisition found: ")
        print(self.global_best_position)
        print("minimum value :")
        print(self.global_best_value)
    

    def Update_particles_position(self):
        for i in range(pop_size):
            self.Update_particle_position(self.particles[i])
            self.Update_particle_value(self.particles[i])
            self.Update_particles_best_and_global_best_position(self.particles[i])
    
    def Update_particles_velicity(self):
        for i in range(pop_size):
            self.Update_particle_velicity(self.particles[i])
        
    def Update_particle_position(self, particle):
        for dim in range(num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0] * 0.9
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1] * 0.9
    
    def Update_particle_value(self, particle):
        particle.value = self.function(particle.position)
                    
    def Update_particle_velicity(self, particle):
        for dim in range(num_var):
                particle.velocity[dim] = self.w * particle.velocity[dim] \
                    + self.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
        
    def Update_particles_best_and_global_best_position(self, particle):
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value.copy()
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value.copy()

    def Update_inertia_weight(self):
        self.w = 0.5 + np.random.random()/2


def test_funcion(x):
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\

num_var = 6
pop_size = 10
c1 = 0.8
c2 = 0.6
epochs = 100 
range_of_params = [(-10,10)]*num_var

pso = PSO(num_var,pop_size,c1,c2,epochs,range_of_params)
pso.Start(test_funcion);



