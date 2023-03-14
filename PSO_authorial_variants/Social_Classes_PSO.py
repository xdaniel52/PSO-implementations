import numpy as np
import math
import xml.etree.ElementTree as gfg 
from matplotlib import pyplot as plt
import sys
sys.path.append('.')  
from Test_templates.test_functions import *

class Particle:
     "Represents all information about one partile"
     def __init__(self, position, value, velocity):
        self.position = position
        self.value = value
        self.velocity = velocity
        self.best_position = position
        self.best_value = value    
    
class UpperMiddleClass:
    def __init__(self,params) -> None:
        self.pop_size = params[0]
        self.w = params[1]
        self.c1 = params[2]
        self.c2 = params[3]
        self.particles = []
class LowerClass(UpperMiddleClass):
    def __init__(self,params) -> None:
        super().__init__(params[:4])
        self.neighborhood_size = params[4]
        self.lbest_value = 0
        self.lbest_position = 0
        self.particles = []

class PSO:
    def __init__(self,upper_params: list,middle_params: list,lower_params: list,generation_change: int
                 ,num_var: int,epochs: int,convergence_generation_params: list = [] ) -> None:

        self.upper = UpperMiddleClass(upper_params)
        self.middle = UpperMiddleClass(middle_params)
        self.lower = LowerClass(lower_params)
        
        self.global_best_position = 0
        self.global_best_value = 0
        self.generation_change = generation_change
        self.generation_change_counter = 0
        self.num_var = num_var
        self.epochs = epochs
        
        self.limit_counter = 0       
        
        if len(convergence_generation_params)==0:
            self.convergence_generation = False
        else:
            self.convergence_generation = True
            self.c_gen_w_max = convergence_generation_params[0]
            self.c_gen_w_min = convergence_generation_params[1] 
            self.c_gen_c1 = convergence_generation_params[2] 
            self.c_gen_c2 = convergence_generation_params[3] 
          
        
    def Init_particles(self) -> None:
        "Initialize all groups"
        self.Init_group(self.upper.particles,self.upper.pop_size)
        self.Init_group(self.middle.particles,self.middle.pop_size)
        self.Init_group(self.lower.particles,self.lower.pop_size)
            
    def Init_group(self,particles: list,size: int) -> None:
        "Initialize given list with particles"
        particles[:] = []
        for i in range(size):
            position = []
            velocity = []
            for dim in range(self.num_var):
                position.append(np.random.random()*(
                    self.range_of_params[dim][1]-self.range_of_params[dim][0]) + self.range_of_params[dim][0]) 
                velocity.append(0.1 * position[dim]) 
            value = self.function(position)
            particles.append(Particle(position, value, velocity, self.num_var))
    
    def Start(self, function, range_of_params: list, plot_gbest: bool = False, plot_velocity: bool  = False) -> None:
        "Starts algorithm and search for optimum"
        self.function = function
        self.range_of_params = range_of_params
        self.velocity_bot_limit = range_of_params[0][0]/10
        self.velocity_top_limit = range_of_params[0][1]/10
        
        self.Init_particles()
        self.Find_starting_global_best()
        self.Plots_before_start(plot_gbest,plot_velocity)
        self.Plots_update_history(plot_gbest,plot_velocity)
        self.Convergence_generation_before_epochs()
                    
        for e in range(self.epochs):
            self.Update_particles_position()
            self.Update_particles_velicity()
            self.Plots_update_history(plot_gbest,plot_velocity)
            if self.generation_change_counter >= self.generation_change:
                self.Update_classes()
                self.generation_change_counter = 0
            else:
                self.generation_change_counter+=1
                
        self.Convergence_generation_after_epochs(plot_gbest,plot_velocity)
        
        self.Plots_plot(plot_gbest,plot_velocity)              
        #self.Print_result()
    
    def Find_starting_global_best(self) -> None:
        "Search for best value in new particles"
        self.global_best_position = self.upper.particles[0].position.copy()
        self.global_best_value = self.upper.particles[0].value
        for i in range(self.upper.pop_size):
            if(self.upper.particles[i].value < self.global_best_value):
                self.global_best_position = self.upper.particles[i].position.copy()
                self.global_best_value = self.upper.particles[i].value
        for i in range(self.middle.pop_size):
            if(self.middle.particles[i].value < self.global_best_value):
                self.global_best_position = self.middle.particles[i].position.copy()
                self.global_best_value = self.middle.particles[i].value
        for i in range(self.lower.pop_size):
            if(self.lower.particles[i].value < self.global_best_value):
                self.global_best_position = self.lower.particles[i].position.copy()
                self.global_best_value = self.lower.particles[i].value    
        
    def Update_classes(self) -> None:
        "Rearange particles in groups based on their value"
        self.lower.particles.sort(key=lambda p: p.value)
        tmp_list = self.upper.particles + self.middle.particles+self.lower.particles[:self.lower.pop_size]
        tmp_list.sort(key=lambda p: p.value)
        self.upper.particles = tmp_list[:self.upper.pop_size]
        self.middle.particles = tmp_list[self.upper.pop_size:self.upper.pop_size+self.middle.pop_size]
        self.lower.particles = tmp_list[self.upper.pop_size+self.middle.pop_size:]
        
            
    def Print_result(self) -> None:
        "Show global best value and position"
        print(f"best pisition found: {self.global_best_position}")
        print(f"minimum value : {self.global_best_value}")
    

    def Update_particles_position(self) -> None:
        "Update particles in all groups"
        self.Update_particles_position_in_group(self.upper)
        self.Update_particles_position_in_group(self.middle)
        self.Update_particles_position_in_group(self.lower)
    
    def Update_particles_position_in_group(self, group: LowerClass) -> None:
        "Update particles in given group"
        for i in range(group.pop_size):        
            self.Update_particle_position(group.particles[i])       
            self.Update_particle_value(group.particles[i])
            self.Update_particle_best_and_global_best_position(group.particles[i])
            
    def Update_particle_position(self, particle: Particle) -> None:
        "Add particle's velocity to its position"
        for dim in range(self.num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0] * 0.9
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1] * 0.9
    
    def Update_particle_value(self, particle: Particle) -> None:
        "Calculate new value of function for particle's position"
        particle.value = self.function(particle.position) 
        
    def Update_particles_velicity(self) -> None:
        "Update velocity forparticles in each group"
        self.Update_upper_particles_velicity()
        self.Update_middle_particles_velicity()
        self.Update_lower_particles_velicity()
            
    def Update_upper_particles_velicity(self) -> None:
        "Update velocity for each particle in upper population"
        for i in range(self.upper.pop_size):           
            self.Update_upper_particle_velicity(self.upper.particles[i])
            self.CheckVelocityLimits(self.upper.particles[i])
    
    def Update_middle_particles_velicity(self) -> None:
        "Update velocity for each particle in middle population"
        for i in range(self.middle.pop_size):
            self.Update_middle_particle_velicity(self.middle.particles[i])
            self.CheckVelocityLimits(self.middle.particles[i])
            
    def Update_lower_particles_velicity(self) -> None:
        "Update velocity for each particle in lower population"
        for i in range(self.lower.pop_size):   
            self.Find_local_best(i)
            self.Update_lower_particle_velicity(self.lower.particles[i]) 
            self.CheckVelocityLimits(self.lower.particles[i])
                           
    def Update_upper_particle_velicity(self, particle: Particle) -> None: 
        "Update velocity for given upper particle"         
        for dim in range(self.num_var):           
                particle.velocity[dim] = self.upper.w * particle.velocity[dim] \
                    + self.upper.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.upper.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
              
    def Update_middle_particle_velicity(self, particle: Particle) -> None:
        "Update velocity for given middle particle"  
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.middle.w * particle.velocity[dim] \
                    + self.middle.c1 * np.random.random() * \
                            (self.lower.particles[particle.learining_vector[dim]].best_position[dim] - particle.position[dim])\
                    + self.middle.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
    
    def Update_lower_particle_velicity(self, particle: Particle) -> None:
        "Update velocity for given lower particle"  
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.lower.w * particle.velocity[dim] \
                    + self.lower.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.lower.c2 * np.random.random() * (self.lower.lbest_position[dim] - particle.position[dim]) 
      
    def CheckVelocityLimits(self,particle: Particle) -> None:
        "Checks if particle has grater velocity then velocity limit"
        for dim in range(self.num_var): 
            if particle.velocity[dim] < self.velocity_bot_limit:
                #print(particle.velocity[dim], "   ", self.velocity_bot_limit)
                #self.limit_counter+=1             
                particle.velocity[dim] = self.velocity_bot_limit
                
            elif particle.velocity[dim] > self.velocity_top_limit:
                #print(particle.velocity[dim], "   ", self.velocity_top_limit)
                #self.limit_counter+=1
                particle.velocity[dim] = self.velocity_top_limit
                
        
        
    def Update_particle_best_and_global_best_position(self, particle: Particle) -> None:
        "Update best and global best position"  
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value             
            
    def Find_local_best(self, idx: int) -> None: 
        "Finds local best for given index of particle"
        self.lower.lbest_value = self.lower.particles[idx].best_value
        self.lower.lbest_position = self.lower.particles[idx].best_position.copy()
        for i in range(1,self.lower.neighborhood_size // 2 +1):
            down_idx = idx-i
            if self.lower.particles[down_idx].best_value < self.lower.lbest_value:          
                    self.lower.lbest_value = self.lower.particles[down_idx].best_value   
                    self.lower.lbest_position = self.lower.particles[down_idx].best_position.copy() 
                    
            upper_idx = idx+i-self.lower.pop_size if idx+i-self.lower.pop_size >= 0 else idx+i 
            if self.lower.particles[upper_idx].best_value < self.lower.lbest_value:          
                self.lower.lbest_value = self.lower.particles[upper_idx].best_value    
                self.lower.lbest_position = self.lower.particles[upper_idx].best_position.copy() 
    
    def GetOptimum(self) -> float:
        "Returns global best"
        return self.global_best_value
    
    def GetBestPosition(self) -> float:
        "Returns global best"
        return self.global_best_position
    
    def AddElementToTree(self, parent, elem_name, text) -> None:
        "Adds new element to XML tree"
        elementXML = gfg.Element(elem_name) 
        parent.append(elementXML)
        elementXML.text = str(text)
        
    
    def Create_XML_Tree(self, fun_Name: str) -> None:
        "Creates new XML tree"
        testXML = gfg.Element('TEST') 
        
        self.AddElementToTree(parent = testXML,elem_name = 'PSO_NAME',text = "Social Classes PSO")
        self.AddElementToTree(parent = testXML,elem_name = 'FUNCTION',text = fun_Name)
        
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_POP_SIZE',text = self.upper.pop_size)
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_W',text = self.upper.w)
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_C1',text = self.upper.c1)
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_C2',text = self.upper.c2)

        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_POP_SIZE',text = self.middle.pop_size)
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_W',text = self.middle.w)
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_C1',text = self.middle.c1)
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_C2',text = self.middle.c2)
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_REFRESHING_GAP',text = self.middle.refreshing_gap)
        
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_POP_SIZE',text = self.lower.pop_size)
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_W',text = self.lower.w)
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_C1',text = self.lower.c1)
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_C2',text = self.lower.c2)
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_NEIGHBORHOOD_SIZE',text = self.lower.neighborhood_size)
        
        self.AddElementToTree(parent = testXML,elem_name = 'NUM_VAR',text = self.num_var)
        self.AddElementToTree(parent = testXML,elem_name = 'GENERATION_CHANGE',text = self.generation_change)
        self.AddElementToTree(parent = testXML,elem_name = 'EPOCHS',text = self.epochs)
        
        self.AddElementToTree(parent = testXML,elem_name = 'GLOBAL_BEST',text = self.global_best_value)
        
        return testXML
   
    def ExportResultToXML(self, file_Name: str, fun_Name: str) -> None:
        "Returns global best"
        root = gfg.Element('ROOT')   
        testXML = self.Create_XML_Tree(fun_Name)
        root.append(testXML)
        tree = gfg.ElementTree(root)     
        with open (file_Name+".xml","wb") as file :
            tree.write(file)
            
    def Plots_before_start(self, plot_gbest: bool,plot_velocity: bool) -> None:
        "Inits plots variables"
        if plot_gbest:
            self.global_best_history = []
            self.upper_best_history = []
            self.middle_best_history = []
            self.lower_best_history = []
            
        if plot_velocity:
            self.global_velocity_history = []
            self.upper_velocity_history = []
            self.middle_velocity_history = []
            self.lower_velocity_history = []
            
    def Plots_update_history(self, plot_gbest: bool,plot_velocity: bool) -> None:
        "Update plots info"
        if plot_gbest:
            self.global_best_history.append(self.global_best_value)
            self.upper_best_history.append(sum([p.best_value for p in self.upper.particles])/self.upper.pop_size)
            self.middle_best_history.append(sum([p.best_value for p in self.middle.particles])/self.middle.pop_size)
            self.lower_best_history.append(sum([p.best_value for p in self.lower.particles])/self.lower.pop_size)
            
        if plot_velocity:           
            self.upper_velocity_history.append(\
                sum([sum(p.velocity[:5])/5 for p in self.upper.particles])/self.upper.pop_size)
            self.middle_velocity_history.append(\
                sum([sum(p.velocity[:5])/5 for p in self.middle.particles])/self.middle.pop_size)
            self.lower_velocity_history.append(\
                sum([sum(p.velocity[:5])/5 for p in self.lower.particles])/self.lower.pop_size)
            self.global_velocity_history.append((
                self.upper_velocity_history[-1]*self.upper.pop_size+
                self.middle_velocity_history[-1]*self.middle.pop_size+
                self.lower_velocity_history[-1]*self.lower.pop_size)/
                (self.upper.pop_size+self.middle.pop_size+self.lower.pop_size))
            
    def Plots_plot(self, plot_gbest: bool,plot_velocity: bool) -> None: 
        "Plots plots"
        if plot_gbest:
            plt.plot(self.global_best_history[:],label="global")
            plt.plot(self.upper_best_history[:],label="upper")
            plt.plot(self.middle_best_history[:],label="middle")
            plt.plot(self.lower_best_history[:],label="lower")
            plt.xlabel("Epochs")
            plt.ylabel("Global best")
            plt.title("Global best during epochs " + self.function.__name__)
            plt.legend()
            plt.show()
        if plot_velocity:           
            plt.plot(self.global_velocity_history[:],label="global")
            plt.plot(self.upper_velocity_history[:],label="upper")
            plt.plot(self.middle_velocity_history[:],label="middle")
            plt.plot(self.lower_velocity_history[:],label="lower")
            plt.xlabel("Epochs")
            plt.ylabel("avg velocity")
            plt.title("Average velocity during epochs " + self.function.__name__)
            plt.legend()
            plt.show()
            
    def Calculate_inertia_weight(self, epoch, epochs, w_max, w_min) -> None:
        "Calculate dynamic inertia weight for chosen epoch"
        return w_min + (w_max-w_min)*(epochs - epoch)/epochs
        

def test_funcion(x: list) -> float:
    return   10*np.power(x[0]-1.,2)\
            +20*np.power(x[1]-2.,2)\
            +30*np.power(x[2]-3.,2)\
            +40*np.power(x[3]-4.,2)\
            +50*np.power(x[4]-5.,2)\
            +60*np.power(x[5]-6.,2)\

def main() -> None:
    #uppper
    pop_size = 5
    w = 0.7
    c1 = 2.0
    c2 = 1.1
    upper_params = (pop_size,w,c1,c2)

    #middle
    pop_size = 5
    w = 0.5
    c1 = 1.1
    c2 = 1.7
    pc = 0.3
    refreshing_gap = 9

    middle_params = (pop_size,w,c1,c2,pc,refreshing_gap)

    #lower
    pop_size = 30
    w = 0.3
    c1 = 1.35
    c2 = 1.6
    neighborhood_size = 5
    lower_params = (pop_size,w,c1,c2,neighborhood_size)

    num_var = 10
    generation_change = 3
    epochs = 1000 

    w_max = 0.9
    w_min = 0.3
    c1 = 0.4
    c2 = 1.8
    convergence_generation_params = (w_max,w_min,c1,c2)

    range_of_params = [(-500,500)]*num_var

    pso = PSO(upper_params,middle_params,lower_params,generation_change,num_var,epochs)
    pso.Start(f_7,range_of_params);
    pso.Print_result()
    
if __name__ == "__main__":
    main()   
        
