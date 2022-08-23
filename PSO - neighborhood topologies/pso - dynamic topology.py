import numpy as np

class Particle:
     def __init__(self, position, value, velocity, num_var):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value
        self.learining_vector = [ 0 for i in range(num_var)]
        self.refreshing_counter = 100


class PSO:
    def __init__(self, num_var, pop_size,w_min,w_max, c1,c2, pc, refreshing_gap, regrouping_period, epochs, range_of_params):
        self.num_var = num_var
        self.pop_size = pop_size
        self.w_min = w_min
        self.w_max = w_max
        self.w = w_max
        self.c1 = c1
        self.c2 = c2
        self.pc = pc
        self.refreshing_gap = refreshing_gap
        self.regrouping_period = regrouping_period
        self.regrouping_counter = 0
        self.lbest_value = 0
        self.lbest_position = 0
        self.epochs = epochs
        self.range_of_params = range_of_params
        self.particles = []
        self.global_best_position = 0
        self.global_best_value = 0
        
    def Init_particles(self):
        self.particles = []
        for i in range(self.pop_size):
            position = []
            velocity = []
            for dim in range(self.num_var):
                position.append(np.random.random()*(self.range_of_params[dim][1]-self.range_of_params[dim][0]) + self.range_of_params[dim][0]) 
                velocity.append(0.1 * position[dim]) 
            value = self.function(position)
            self.particles.append(Particle(position, value, velocity, num_var))
    
    def Start(self, function):
        self.function = function
        self.Init_particles()
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_value = self.particles[0].value
        for i in range(self.pop_size):
            if(self.particles[i].value < self.global_best_value):
                self.global_best_position = self.particles[i].position.copy()
                self.global_best_value = self.particles[i].value 
               
        for e in range(self.epochs):
            self.Update_inertia_weight(e)
            self.Update_particles_position()
            self.Update_particles_velicity()
            self.Regrouping()
                       
        self.Print_result()
    
    def Print_result(self):
        global_best_position, global_best_value = self.Find_best()
        print("best pisition found: ")
        print(global_best_position)
        print("minimum value :")
        print(global_best_value)
    

    def Update_particles_position(self):
        for i in range(self.pop_size):        
            self.Update_particle_position(self.particles[i])       
            self.Update_particle_value(self.particles[i])
            self.Update_particle_best_and_global_best_position(self.particles[i])
    
    def Update_particles_velicity(self):
        for i in range(self.pop_size):
            if self.particles[i].refreshing_counter > self.refreshing_gap:
                self.Refresh_learining_vector(i)
            self.Update_particle_velicity(self.particles[i])
        
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
                    + self.c1 * np.random.random() * \
                    (self.particles[particle.learining_vector[dim]].best_position[dim] - particle.position[dim])\
                    + self.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 

        
    def Update_particle_best_and_global_best_position(self, particle):
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value
            particle.refreshing_counter = 0
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value.copy()
        else:
            particle.refreshing_counter += 1
                 
            
    def Refresh_learining_vector(self, idx): 
        for dim in range(self.num_var):
            if np.random.random() <= self.pc:
                one = np.random.randint(0,self.num_var)
                while one == idx:
                    one = np.random.randint(0,self.num_var)
                    
                two = np.random.randint(0,self.num_var)
                while two == idx:
                    two = np.random.randint(0,self.num_var)
                    
                if self.particles[one].best_value < self.particles[two].best_value:
                    self.particles[idx].learining_vector[dim] = one
                else:
                    self.particles[idx].learining_vector[dim] = two
            else:
                self.particles[idx].learining_vector[dim] = idx
                
        self.particles[idx].refreshing_counter = 0
    
    def Update_inertia_weight(self, epoch):
        self.w = self.w_min + (self.w_max-self.w_min)*(self.epochs - epoch)/self.epochs
        
    def Regrouping(self):
        if self.regrouping_counter >= self.regrouping_period:
            new_list = []
            index_list = np.random.permutation(self.pop_size)
            for i in range(self.pop_size):
                new_list.append(self.particles[index_list[i]])
            self.particles = new_list
            self.regrouping_counter = 0    
        else:
            self.regrouping_counter += 1
    
    def Find_best(self):
        best_idx = 0
        for i in range(1,self.pop_size):
            if(self.particles[i].best_value < self.particles[best_idx].best_value):
                best_idx = i
                
        return (self.particles[best_idx].best_position,self.particles[best_idx].best_value)
    
                     

def test_funcion(x):
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\

num_var = 6
pop_size = 10
w_min = 0.4
w_max = 0.9
c1 = 1.49
c2 = 1.49
pc = 0.3
refreshing_gap = 3
regrouping_period = 5
epochs = 500
range_of_params = [(-10,10)]*num_var

pso = PSO(num_var,pop_size,w_min,w_max,c1,c2,pc,refreshing_gap,regrouping_period,epochs,range_of_params)
pso.Start(test_funcion);


