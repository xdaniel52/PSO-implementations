import numpy as np

class Particle:
     def __init__(self, position: list, value : float, velocity: list):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value


class PSO:
    def __init__(self, num_var: int, pop_size: int, w: float, c1_start: float, c1_end: float, c2_start: float, c2_end: float, epochs: int ):
        self.num_var = num_var
        self.pop_size = pop_size
        self.w = w
        self.c1_start = c1_start
        self.c1_end = c1_end
        self.c2_start = c2_start
        self.c2_end = c2_end
        self.c1 = c1_start
        self.c2 = c2_start
        self.epochs = epochs
        
        self.global_best_position = 0
        self.global_best_value = 0
        self.particles = []
        
    def Init_particles(self) -> None:
        self.particles = []
        for i in range(self.pop_size):
            position = []
            velocity = []
            for dim in range(self.num_var):
                position.append(np.random.random()*(self.range_of_params[dim][1]-self.range_of_params[dim][0]) + self.range_of_params[dim][0]) 
                velocity.append(0.1 * position[dim]) 
            value = self.function(position)
            self.particles.append(Particle(position, value, velocity))
    
    def Start(self, function: function, range_of_params: list) -> None:
        self.range_of_params = range_of_params
        self.function = function
        self.Init_particles()
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_value = self.particles[0].value.copy()
        for i in range(self.pop_size):
            if(self.particles[i].value < self.global_best_value):
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_value = self.particles[i].value.copy() 
               
        for epoch in range(self.epochs):  
            self.Update_acceleration_coefficients(epoch)
            self.Update_particles_velicity()
            self.Update_particles_position()                     
    
    def Print_result(self):
        print(f"best pisition found: {self.global_best_position}")
        print(f"minimum value : {self.global_best_value}")
    

    def Update_particles_position(self) -> None:
        for i in range(self.pop_size):
            self.Update_particle_position(self.particles[i])
            self.Update_particle_value(self.particles[i])
            self.Update_particles_best_and_global_best_position(self.particles[i])
    
    def Update_particles_velicity(self) -> None:
        for i in range(self.pop_size):
            self.Update_particle_velicity(self.particles[i])
        
    def Update_particle_position(self, particle: Particle) -> None:
        for dim in range(self.num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0] * 0.9
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1] * 0.9
    
    def Update_particle_value(self, particle: Particle) -> None:
        particle.value = self.function(particle.position)
                    
    def Update_particle_velicity(self, particle: Particle) -> None:
        for dim in range(self.num_var):
                particle.velocity[dim] = self.w * particle.velocity[dim] \
                    + self.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
        
    def Update_particles_best_and_global_best_position(self, particle: Particle) -> None:
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value.copy()
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value.copy()

    def Update_acceleration_coefficients(self, epoch: int) -> None:
        self.c1 = self.c1_start + (self.c1_end-self.c1_start)*(epoch/self.epochs)
        self.c2 = self.c2_start + (self.c2_end-self.c2_start)*(epoch/self.epochs)

def test_funcion(x: list) -> float:
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\

def main():
    num_var = 6
    pop_size = 10
    w = 0.8
    c1_start = 0.5
    c1_end =  2.5
    c2_start = 2.5
    c2_end =  0.5
    epochs = 100 
    range_of_params = [(-10,10)]*num_var

    pso = PSO(num_var,pop_size,w,c1_start,c1_end,c2_start,c2_end,epochs)
    pso.Start(test_funcion,range_of_params)
    pso.Print_result()

if __name__ == "__main__":
    main()
