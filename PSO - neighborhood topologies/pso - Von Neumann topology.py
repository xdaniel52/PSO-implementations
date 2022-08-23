import numpy as np

class Particle:
     def __init__(self, position, value, velocity):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value


class PSO:
    def __init__(self, num_var, pop_shape, w, c1, c2, epochs, range_of_params):
        self.num_var = num_var
        self.pop_shape = pop_shape
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.lbest_value = 0
        self.lbest_position = 0
        self.epochs = epochs
        self.range_of_params = range_of_params
        self.particles = []
        
    def Init_particles(self):
        self.particles = []
        for i in range(self.pop_shape[0]):
            self.particles.append([])
            for j in range(self.pop_shape[1]):
                
                position = []
                velocity = []
                for dim in range(self.num_var):
                    position.append(np.random.random()*(self.range_of_params[dim][1]-self.range_of_params[dim][0]) + self.range_of_params[dim][0]) 
                    velocity.append(0.1 * position[dim]) 
                value = self.function(position)
                self.particles[-1].append(Particle(position, value, velocity))
    
    def Start(self, function):
        self.function = function
        self.Init_particles()
               
        for e in range(self.epochs):
            self.Update_particles_position()
            self.Update_particles_velicity()       
                       
        self.Print_result()
    
    def Print_result(self):
        global_best_position, global_best_value = self.Find_best()
        print("best pisition found: ")
        print(global_best_position)
        print("minimum value :")
        print(global_best_value)
    

    def Update_particles_position(self):
        for i in range(self.pop_shape[0]):
            for j in range(self.pop_shape[1]):       
                self.Update_particle_position(self.particles[i][j])       
                self.Update_particle_value(self.particles[i][j])
                self.Update_particle_best(self.particles[i][j])
    
    def Update_particles_velicity(self):
        for i in range(self.pop_shape[0]):
            for j in range(self.pop_shape[1]):
                self.Find_local_best(i,j)
                self.Update_particle_velicity(self.particles[i][j])
        
    def Update_particle_position(self, particle):
        for dim in range(self.num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0] * 0.9
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1] * 0.9
    
    def Update_particle_value(self, particle):
        particle.value = self.function(particle.position)
                    
    def Update_particle_velicity(self, particle):
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.w * particle.velocity[dim] \
                    + self.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.c2 * np.random.random() * (self.lbest_position[dim] - particle.position[dim]) 
        
    def Update_particle_best(self, particle):
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value
    def Find_local_best(self, row_idx,col_idx): 
        
        self.lbest_value = self.particles[row_idx][col_idx].best_value
        self.lbest_position = self.particles[row_idx][col_idx].best_position.copy()

        if self.particles[row_idx-1][col_idx].best_value < self.lbest_value:          
                self.lbest_value = self.particles[row_idx-1][col_idx].best_value   
                self.lbest_position = self.particles[row_idx-1][col_idx].best_position.copy() 
                
        if self.particles[row_idx][col_idx-1].best_value < self.lbest_value:          
            self.lbest_value = self.particles[row_idx][col_idx-1].best_value   
            self.lbest_position = self.particles[row_idx][col_idx-1].best_position.copy()
                
        row_idx_up = row_idx+1-self.pop_shape[0] if row_idx+1-self.pop_shape[0] >= 0 else row_idx+1 
        
        if self.particles[row_idx_up][col_idx].best_value < self.lbest_value:          
            self.lbest_value = self.particles[row_idx_up][col_idx].best_value    
            self.lbest_position = self.particles[row_idx_up][col_idx].best_position.copy() 
            
        col_idx_up = col_idx+1-self.pop_shape[1] if col_idx+1-self.pop_shape[1] >= 0 else col_idx+1  
        
        if self.particles[row_idx][col_idx_up].best_value < self.lbest_value:          
            self.lbest_value = self.particles[row_idx][col_idx_up].best_value    
            self.lbest_position = self.particles[row_idx][col_idx_up].best_position.copy()
         
    def Find_best(self):
        best_idx = (0,0)
        for i in range(self.pop_shape[0]):
            for j in range(self.pop_shape[1]):
                if(self.particles[i][j].best_value < self.particles[best_idx[0]][best_idx[1]].best_value):
                    best_idx = (i,j)
                
        return (self.particles[best_idx[0]][best_idx[1]].best_position,self.particles[best_idx[0]][best_idx[1]].best_value)
                     

def test_funcion(x):
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\

num_var = 6
pop_size = 10
pop_shape = (3,4)
w = 0.8
c1 = 0.8
c2 = 0.6
epochs = 100
range_of_params = [(-10,10)]*num_var

pso = PSO(num_var,pop_shape,w,c1,c2,epochs,range_of_params)
pso.Start(test_funcion);


