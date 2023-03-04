import numpy as np
import math
import xml.etree.ElementTree as gfg 
from matplotlib import pyplot as plt
import sys
sys.path.append('.')  
from Test_templates.test_functions import *

class Particle:
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
        self.num_var = num_var
        self.epochs = epochs
        
        self.cPop = []
        self.helper_value = 0
        self.nc = 2
        self.k = 3
        self.helperPop = []
        self.cpop_size = 0
        self.generation_change = generation_change
        
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
        self.Init_group(self.upper.particles,self.upper.pop_size)
        self.Init_group(self.middle.particles,self.middle.pop_size)
        self.Init_group(self.lower.particles,self.lower.pop_size)
      
    def Init_group(self,particles: list,size: int) -> None:
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
    
    def Start(self, function, range_of_params: list, plot_gbest: bool = False,
               plot_velocity: bool = False,plot_best_worst: bool = False) -> None:
        self.function = function
        self.range_of_params = range_of_params
        self.velocity_bot_limit = range_of_params[0][0]/10
        self.velocity_top_limit = range_of_params[0][1]/10
        self.helper_value = (range_of_params[0][1] - range_of_params[0][0])/20  
        self.Init_particles()
        self.Find_starting_global_best()
        self.Plots_before_start(plot_gbest,plot_velocity,plot_best_worst)
        self.Plots_update_history(plot_gbest,plot_velocity,plot_best_worst)
        self.Convergence_generation_before_epochs()
                    
        for e in range(self.epochs):         
            self.cPop = []
            self.helperPop = self.upper.particles +self.middle.particles +self.lower.particles 
            self.Update_particles_position_surrogate()  
            self.Generate_new_particles(e)
            self.Chose_particles_from_cPop()
            self.Update_particles_bests_value()
            self.Update_particles_velicity()
            self.Plots_update_history(plot_gbest,plot_velocity,plot_best_worst)
   
        self.Convergence_generation_after_epochs(plot_gbest,plot_velocity,plot_best_worst)
        self.Plots_plot(plot_gbest,plot_velocity,plot_best_worst)              
        #self.Print_result()
    
    def Find_starting_global_best(self) -> None:
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
        
        
            
    def Print_result(self) -> None:
        print(f"best pisition found: {self.global_best_position}")
        print(f"minimum value : {self.global_best_value}")
    

    def Chose_particles_from_cPop(self) -> None:
        self.cPop.sort(key=lambda p: p.value)

        for i in range(self.upper.pop_size):        
            candidate_id = 1
            candidate_distance = 1e100
            for j in range(1,len(self.cPop)):
                dis = 0
                for d in range(self.num_var):
                    dis+= pow(self.cPop[0].position[d] - self.cPop[j].position[d],2)
                dis = math.sqrt(dis)
                if dis < candidate_distance:
                    candidate_id = j
                    candidate_distance = dis
                    
            self.cPop.pop(candidate_id)
            self.upper.particles[i] = self.cPop[0]
            self.cPop.pop(0)
            self.Update_particle_value(self.upper.particles[i])  
        
        for i in range(self.middle.pop_size):        
            candidate_id = 1
            candidate_distance = 1e100
            for j in range(1,len(self.cPop)):
                dis = 0
                for d in range(self.num_var):
                    dis+= pow(self.cPop[0].position[d] - self.cPop[j].position[d],2)
                dis = math.sqrt(dis)
                if dis < candidate_distance:
                    candidate_id = j
                    candidate_distance = dis
                    
            self.cPop.pop(candidate_id)
            self.middle.particles[i] = self.cPop.pop(0)
            self.Refresh_learining_vector(i)
            self.Update_particle_value(self.middle.particles[i]) 
            
        for i in range(self.lower.pop_size):        
            candidate_id = 1
            candidate_distance = 1e100
            for j in range(1,len(self.cPop)):
                dis = 0
                for d in range(self.num_var):
                    dis+= pow(self.cPop[0].position[d] - self.cPop[j].position[d],2)
                dis = math.sqrt(dis)
                if dis < candidate_distance:
                    candidate_id = j
                    candidate_distance = dis
                    
            self.cPop.pop(candidate_id)
            self.lower.particles[i] = self.cPop.pop(0)
            self.Update_particle_value(self.lower.particles[i])
                
    
    def Update_particles_position_surrogate(self) -> None:
        self.Update_particles_position_in_group_surrogate(self.upper)
        self.Update_particles_position_in_group_surrogate(self.middle)
        self.Update_particles_position_in_group_surrogate(self.lower)
        
        
    def Update_particles_position_in_group_surrogate(self, group: LowerClass) -> None:
        for i in range(group.pop_size): 
            particle = Particle(0,0,0,self.num_var)
            particle.Init2(group.particles[i])
            self.Update_particle_position(particle)       
            self.Calculate_value_new_particle_surrogate(particle)
            #self.Update_particle_value(particle)
            self.cPop.append(particle)
            
    def Generate_new_particles(self,epoch: int) -> None:
        self.cpop_size = len(self.cPop)
        self.helper_value=((self.range_of_params[0][1]-self.range_of_params[0][0])/20)*((1.0-epoch/self.epochs)**2)
        for j in range(self.nc):
            for i in range(self.cpop_size):
                particle = Particle(0,0,0,self.num_var)
                particle.Init2(self.cPop[i])
                self.Change_new_particle(particle, epoch)
                self.Calculate_value_new_particle_surrogate(particle)
                #self.Update_particle_value(particle)
                self.cPop.append(particle)
                    
    def Update_particle_value(self, particle: Particle) -> None:
        particle.value = self.function(particle.position)            
            
    def Change_new_particle(self,particle: Particle,epoch: int) -> None:
        for dim in range(self.num_var):
            prob = 0.05+0.10*(1 - epoch/self.epochs)
            random = np.random.random()
            if random < prob:
                particle.position[dim] += np.random.random()*particle.velocity[dim]*3
                if particle.velocity[dim] > 0:
                   particle.position[dim]+=self.helper_value
                else:
                    particle.position[dim]-=self.helper_value
                
                if(particle.position[dim] < self.range_of_params[dim][0]):
                    particle.position[dim] = self.range_of_params[dim][0] * 0.9
                elif(particle.position[dim] > self.range_of_params[dim][1]):
                    particle.position[dim] = self.range_of_params[dim][1] * 0.9
                
    
    def Calculate_value_new_particle_surrogate(self,particle: Particle) -> None:
        bests = [(2e20,0) for o in range(3)]
        for i in range(len(self.helperPop)):
            dis = 0.0
            for d in range(self.num_var):
                dis+= pow(particle.position[d] - self.helperPop[i].position[d],2)
            dis = math.sqrt(dis)
            if dis < bests[2][0]:
                bests[0] = bests[1]
                bests[1] = bests[2]
                bests[2] = (dis, self.helperPop[i].value)
            elif dis < bests[1][0]:
                bests[0] = bests[1]
                bests[1] = (dis, self.helperPop[i].value)
            elif dis < bests[0][0]:
                bests[0] = (dis, self.helperPop[i].value)
                
        particle.value = (bests[0][1]+bests[1][1]+bests[2][1])/3
         
    def Update_particles_bests_value(self) -> None:
        for i in range(self.upper.pop_size):
            self.Update_particle_bests_value(self.upper.particles[i])
        for i in range(self.middle.pop_size):
            self.Update_particle_bests_value(self.middle.particles[i])
        for i in range(self.lower.pop_size):
            self.Update_particle_bests_value(self.lower.particles[i])
        
    def Update_particle_bests_value(self,particle) -> None:
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value
            particle.refreshing_counter = 0
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value             
          
    def Update_particle_position(self, particle: Particle) -> None:
        for dim in range(self.num_var):
            particle.position[dim] += particle.velocity[dim]
            if(particle.position[dim] < self.range_of_params[dim][0]):
                particle.position[dim] = self.range_of_params[dim][0] * 0.9
            elif(particle.position[dim] > self.range_of_params[dim][1]):
                particle.position[dim] = self.range_of_params[dim][1] * 0.9
        
    def Update_particles_velicity(self) -> None:
        self.Update_upper_particles_velicity()
        self.Update_middle_particles_velicity()
        self.Update_lower_particles_velicity()
            
    def Update_upper_particles_velicity(self) -> None:
        for i in range(self.upper.pop_size):           
            self.Update_upper_particle_velicity(self.upper.particles[i])
            self.CheckVelocityLimits(self.upper.particles[i])
    
    def Update_middle_particles_velicity(self) -> None:
        for i in range(self.middle.pop_size):
            self.Update_middle_particle_velicity(self.middle.particles[i])
            self.CheckVelocityLimits(self.middle.particles[i])
            
    def Update_lower_particles_velicity(self) -> None:
        for i in range(self.lower.pop_size):   
            self.Find_local_best(i)
            self.Update_lower_particle_velicity(self.lower.particles[i]) 
            self.CheckVelocityLimits(self.lower.particles[i])
                           
    def Update_upper_particle_velicity(self, particle: Particle) -> None:         
        for dim in range(self.num_var):    
                particle.velocity[dim] = self.upper.w * particle.velocity[dim] \
                    + self.upper.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.upper.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
                        
                    
    def Update_middle_particle_velicity(self, particle: Particle) -> None:
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.middle.w * particle.velocity[dim] \
                    + self.middle.c1 * np.random.random() * \
                            (self.lower.particles[particle.learining_vector[dim]].position[dim] - particle.position[dim])\
                    + self.middle.c2 * np.random.random() * (self.global_best_position[dim] - particle.position[dim]) 
    
    def Update_lower_particle_velicity(self, particle: Particle) -> None:
        for dim in range(self.num_var):            
                particle.velocity[dim] = self.lower.w * particle.velocity[dim] \
                    + self.lower.c1 * np.random.random() * (particle.best_position[dim] - particle.position[dim]) \
                    + self.lower.c2 * np.random.random() * (self.lower.lbest_position[dim] - particle.position[dim]) 
      
    def CheckVelocityLimits(self,particle: Particle) -> None:
        for dim in range(self.num_var): 
            if particle.velocity[dim] < self.velocity_bot_limit:            
                particle.velocity[dim] = self.velocity_bot_limit
                
            elif particle.velocity[dim] > self.velocity_top_limit:
                particle.velocity[dim] = self.velocity_top_limit
                
    def Update_particle_best_and_global_best_position(self, particle: Particle) -> None:
        if(particle.value < particle.best_value):
            particle.best_position = particle.position.copy()
            particle.best_value = particle.value
            particle.refreshing_counter = 0
            if(particle.value < self.global_best_value):
                self.global_best_position = particle.position.copy()
                self.global_best_value = particle.value             
            
    def Find_local_best(self, idx: int) -> None: 
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
         
    
    def Refresh_learining_vector(self, idx: int) -> None:  
        for dim in range(self.num_var):
            one = np.random.randint(0,self.lower.pop_size)                   
            two = np.random.randint(0,self.lower.pop_size)
            while two == one:
                two = np.random.randint(0,self.lower.pop_size)
                
            if self.lower.particles[one].best_value < self.lower.particles[two].best_value:
                self.middle.particles[idx].learining_vector[dim] = one
            else:
                self.middle.particles[idx].learining_vector[dim] = two
                
        self.middle.particles[idx].refreshing_counter = 0
    
    def GetOptimum(self) -> float:
        return self.global_best_value
    
    def GetBestPosition(self) -> float:
        return self.global_best_position
    
    def AddElementToTree(self, parent, elem_name, text) -> None:
        elementXML = gfg.Element(elem_name) 
        parent.append(elementXML)
        elementXML.text = str(text)
        
    
    def Create_XML_Tree(self, fun_Name) -> None:
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
   
    def ExportResultToXML(self, file_Name, fun_Name) -> None:
        root = gfg.Element('ROOT')   
        testXML = self.Create_XML_Tree(fun_Name)
        root.append(testXML)
        tree = gfg.ElementTree(root)     
        with open (file_Name+".xml","wb") as file :
            tree.write(file)
            
    def Plots_before_start(self, plot_gbest: bool,plot_velocity: bool,plot_best_worst: bool) -> None:
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
            
        if plot_best_worst:
            self.best_history = []
            self.worst_history = []
            
    def Plots_update_history(self, plot_gbest: bool,plot_velocity: bool,plot_best_worst: bool) -> None:
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
            
        if plot_best_worst:
            self.best_history.append(self.upper.particles[0].value)
            self.worst_history.append(self.lower.particles[-1].value)
            
    def Plots_plot(self, plot_gbest: bool,plot_velocity: bool,plot_best_worst: bool) -> None: 
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
        if plot_best_worst:
            plt.plot(self.best_history[:],label="best")
            plt.plot(self.worst_history[:],label="worst")
            plt.xlabel("Epochs")
            plt.ylabel("value")
            plt.title("Best and worst particles value over epochs in fun " + self.function.__name__)
            plt.legend()
            plt.show()
                  
    def Calculate_inertia_weight(self, epoch: int, epochs: int, w_max: float, w_min: float) -> float:
        return w_min + (w_max-w_min)*(epochs - epoch)/epochs

        
    def Convergence_generation_before_epochs(self) -> None:
        if self.convergence_generation:
            self.c_gen_epochs = int(self.epochs*0.1)
            self.epochs = int(self.epochs*0.9)
     
    def Convergence_generation_after_epochs(self, plot_gbest: bool,plot_velocity: bool,plot_best_worst: bool) -> None:
        if self.convergence_generation:
            self.upper.particles+=self.middle.particles+self.lower.particles          
            self.upper.c1 = self.c_gen_c1
            self.upper.c2 = self.c_gen_c2   
            
            for epoch in range(self.c_gen_epochs):
                self.upper.w = self.Calculate_inertia_weight(epoch, self.c_gen_epochs, self.c_gen_w_max, self.c_gen_w_min )
                self.Update_particles_velicity()
                for particle in self.upper.particles:
                    self.Update_particle_position(particle)
                    self.Update_particle_best_and_global_best_position(particle)
                    
                self.Plots_update_history(plot_gbest,plot_velocity,plot_best_worst)


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
    w = 0.5
    c1 = 1.0
    c2 = 1.5
    upper_params = (pop_size,w,c1,c2)

    #middle
    pop_size = 5
    w = 0.8
    c1 = 2.0
    c2 = 1.5
    pc = 0.3

    middle_params = (pop_size,w,c1,c2,pc)

    #lower
    pop_size = 30
    w = 1.5
    c1 = 1.0
    c2 = 1.8
    neighborhood_size = 5
    lower_params = (pop_size,w,c1,c2,neighborhood_size)

    num_var = 10
    epochs = 1000 

    w_max = 0.9
    w_min = 0.3
    c1 = 0.4
    c2 = 1.8
    convergence_generation_params = (w_max,w_min,c1,c2)

    range_of_params = [(-10,10)]*num_var

    pso = PSO(upper_params,middle_params,lower_params,num_var,epochs)
    pso.Start(f_2,range_of_params,plot_best_worst=True)
    pso.Print_result()
    
if __name__ == "__main__":
    main()      
        
        
        
        
