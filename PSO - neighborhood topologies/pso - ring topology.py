import numpy as np

class Particle:
     def __init__(self, position: list, value : float, velocity: list):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value


class PSO:
    def __init__(self, num_var: int, pop_size: int, w: float, c1: float, c2: float, neighborhood_size: int, epochs: int):
        self.num_var = num_var
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.neighborhood_size = neighborhood_size
        self.lbest_value = 0
        self.lbest_position = 0
        self.epochs = epochs
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
    
    def Start(self, function, range_of_params: list) -> None:
        self.function = function
        self.range_of_params = range_of_params
        self.Init_particles()
               
        for e in range(self.epochs):
            self.Update_particles_position()
            self.Update_particles_velicity()       

    def Print_result(self) -> None:
        global_best_position, global_best_value = self.Find_best()
        print(f"best pisition found: {global_best_position}")
        print(f"minimum value : {global_best_value}")

    def Update_particles_position(self) -> None:
        for i in range(self.pop_size):        
            self.Update_particle_position(self.particles[i])       
            self.Update_particle_value(self.particles[i])
            self.Update_particle_best(self.particles[i])
    
    def Update_particles_velicity(self):
        for i in range(self.pop_size):
            self.Find_local_best(i)
            self.Update_particle_velicity(self.particles[i])
        
    def Update_particle_position(self, particle: Particle) -> None:
        for dim in range(self.num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0]
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1]
    
    def Update_particle_value(self, particle: Particle) -> None:
        particle.value = self.function(particle.position)
                    
    def Update_particle_velicity(self, particle):
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.w * particle.velocity[dim] \
                    + self.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.c2 * np.random.random() * (self.lbest_position[dim] - particle.position[dim]) 
        
    def Update_particle_best(self, particle: Particle) -> None:
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value

    def Find_local_best(self, idx): 
        self.lbest_value = self.particles[idx].best_value
        self.lbest_position = self.particles[idx].best_position.copy()
        for i in range(1,self.neighborhood_size // 2 +1):
            down_idx = idx-i
            if self.particles[down_idx].best_value < self.lbest_value:          
                    self.lbest_value = self.particles[down_idx].best_value   
                    self.lbest_position = self.particles[down_idx].best_position.copy() 
                    
            upper_idx = idx+i-self.pop_size if idx+i-self.pop_size >= 0 else idx+i 
            if self.particles[upper_idx].best_value < self.lbest_value:          
                self.lbest_value = self.particles[upper_idx].best_value    
                self.lbest_position = self.particles[upper_idx].best_position.copy() 
         
    def Find_best(self) -> tuple:
        best_idx = 0
        for i in range(1,self.pop_size):
            if(self.particles[i].best_value < self.particles[best_idx].best_value):
                best_idx = i
                
        return (self.particles[best_idx].best_position,self.particles[best_idx].best_value)
    
    def GetOptimum(self) -> float:
        return self.Find_best()[1]
                     

def test_funcion(x: list) -> float:
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\


def main() -> None:
    num_var = 6
    pop_size = 10
    w = 0.8
    c1 = 0.8
    c2 = 0.6
    neighborhood_size = 3
    epochs = 100 
    range_of_params = [(-100,100)]*num_var

    pso = PSO(num_var,pop_size,w,c1,c2,neighborhood_size,epochs)
    pso.Start(test_funcion, range_of_params)
    pso.Print_result()
    
if __name__ == "__main__":
    main()
