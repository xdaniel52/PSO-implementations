import xml.etree.ElementTree as gfg
import sys
sys.path.append('.')    
from PSO_authorial_variants.Social_Classes_PSO import PSO 
from test_functions import * 

class ParamTuner:
    
    def __init__(self, pso_class, functions_list: list, num_of_tests: int, params_sets_list: list,range_of_params: list):
        self.pso_class = pso_class
        self.functions_list = functions_list
        self.num_of_tests = num_of_tests
        self.params_sets_list = params_sets_list
        self.range_of_params = range_of_params
        lens = [len(par_set)for par_set in self.params_sets_list]
        self.try_count = 1
        for l in lens:
            self.try_count *=l
        self.each_result = [[[0 for i in range(num_of_tests)] for j in functions_list] for k in range(self.try_count)]
        self.ParamsGenerator = self.CreateGenerator()
        self.CalculateNoSteps()

    def Tune(self, verbose_level: int = 1, export: bool = False, file_Name: str = "result", 
             plot_gbest = False,plot_velocity = False ):
        
        if export:
            root = gfg.Element('ROOT')   
            testsXML = gfg.Element('TESTS') 
            root.append(testsXML)        
            
        set_num = 0
        current_step = 0
        for param_set in self.ParamsGenerator:      
            print("Tuning set: ") 
            print(str(param_set))
            for fun_id in range(len(self.functions_list)):
                print(f"    Tuning fun {fun_id}")
                pso = self.pso_class(*param_set) #0
                for test_id in range(self.num_of_tests):             
                    current_step+=1
                    
                    pso.Start(self.functions_list[fun_id],
                              [self.range_of_params[fun_id] for i in range(param_set[4])],
                              plot_gbest,
                              plot_velocity)
                    
                    if export:
                        testXML = pso.Create_XML_Tree(self.functions_list[fun_id].__name__)
                        testsXML.append(testXML)
                    self.each_result[set_num][fun_id][test_id] = pso.GetOptimum()
                    print(f"Progress: {current_step} / {self.steps_number}") 
                   
            set_num+=1
            
        if export:
            tree = gfg.ElementTree(root)     
            with open (file_Name+".xml","wb") as file :
                tree.write(file) 
            
                    
    def CreateGenerator(self):
        lens = [len(par_set)for par_set in self.params_sets_list]
        current_set_ids = [0 for par_set in self.params_sets_list]
        current_set = [par_set[0] for par_set in self.params_sets_list]
        while True:
            for par_set_id in range(len(self.params_sets_list)-1):
                if current_set_ids[par_set_id] == lens[par_set_id]:
                    current_set_ids[par_set_id] = 0
                    current_set_ids[par_set_id+1]+=1
                current_set[par_set_id] = self.params_sets_list[par_set_id][current_set_ids[par_set_id]]
                    
            if current_set_ids[-1] == lens[-1]:
                break
            current_set[-1] = self.params_sets_list[-1][current_set_ids[-1]]  
            yield(current_set.copy())      
            current_set_ids[0]+=1           
            
    def GetGeneratorValue(self, number: int):
        lens = [len(par_set)for par_set in self.params_sets_list]
        current_set_ids = [0 for par_set in self.params_sets_list]
        current_set = [par_set[0] for par_set in self.params_sets_list]
        iterator = 0

        while True:
            for par_set_id in range(len(self.params_sets_list)-1):
                if current_set_ids[par_set_id] == lens[par_set_id]:
                    current_set_ids[par_set_id] = 0
                    current_set_ids[par_set_id+1]+=1
                             
            if iterator == number:
                for par_set_id in range(len(self.params_sets_list)):                   
                    current_set[par_set_id] = self.params_sets_list[par_set_id][current_set_ids[par_set_id]]          
                break   
            iterator+=1  
            current_set_ids[0]+=1                   
        return current_set.copy()
            
    
    def PrintReslt1(self):
        print(f"functions_list: {self.functions_list}")
        print(f"num_of_tests: {self.num_of_tests}")
        print("Results: ")
        self.ParamsGenerator = self.CreateGenerator()
        set_num = 0
        for param_set in self.ParamsGenerator: 
            print(f"Params: {param_set[:-1]}")
            for fun_id in range(len(self.functions_list)): 
                print(f"\t\tfun_id: {fun_id}")
                print(f"\t\t\t\tAVG: {sum(self.each_result[set_num][fun_id])/self.num_of_tests}")
                    
            set_num+=1
                               
    def PrintReslt2(self, num_of_best: int = 2):
        if num_of_best > self.try_count:
            num_of_best = self.try_count
        print("Results: ")   
        for fun_id in range(len(self.functions_list)): 
            tmp_tab = []
            self.ParamsGenerator = self.CreateGenerator()
            for set_num in range(self.try_count): 
                tmp_tab.append((set_num,sum(self.each_result[set_num][fun_id])/self.num_of_tests))
                
                
            tmp_tab.sort(key = lambda e:e[1])    
            print(f"Function: {fun_id} range: {self.range_of_params[fun_id]}")
            print("             [dim, ps, w, c1, c2, epochs]")
            for i in range(num_of_best):
                
                print(f"     best: {i+1} value: {tmp_tab[i][1]}" )
                print(f"     params: {self.GetGeneratorValue(tmp_tab[i][0])}")
      
    def CalculateNoSteps(self):
        steps_number = self.num_of_tests * len(self.functions_list)
        for params_set in self.params_sets_list:
            steps_number*=len(params_set)      
        self.steps_number = steps_number
            
 
def CreateSets(params_sets_list: list):
        '''Return all combination of given params lists '''
        result = []
        lens = [len(par_set)for par_set in params_sets_list]
        current_set_ids = [0 for par_set in params_sets_list]
        current_set = [par_set[0] for par_set in params_sets_list]
        while True:
            for par_set_id in range(len(params_sets_list)-1):
                if current_set_ids[par_set_id] == lens[par_set_id]:
                    current_set_ids[par_set_id] = 0
                    current_set_ids[par_set_id+1]+=1
                current_set[par_set_id] = params_sets_list[par_set_id][current_set_ids[par_set_id]]
                
                  
            if current_set_ids[-1] == lens[-1]:
                break
            current_set[-1] = params_sets_list[-1][current_set_ids[-1]]  
            
            result.append(current_set.copy())      
            current_set_ids[0]+=1 
            
        return result
            
def main():

    functions = [f_1 , f_2 , f_3 , f_4 , f_5 , f_6 , f_7 , f_8 , f_9]
    range_of_params = [(-100,100),(-10,10),(-10,10),(-10,10),(-100,100),(-1.28,1.28),(-500,500),(-5.12,5.12),(-5.12,5.12) ]

    functions = functions[:2]
    range_of_params = range_of_params[:2]
    num_of_tests = 3
    assert len(functions) == len(range_of_params)

    #uppper
    pop_size = [1]
    w = [0.3]
    c1 = [0.5]
    c2 = [2.5] 
    
    upper_params = [pop_size,w,c1,c2]
    upper_params_set=CreateSets(upper_params)

    #middle
    pop_size = [1]
    w = [0.5] 
    c1 = [0.8] 
    c2 = [0.8] 
    refreshing_gap = [3]

    middle_params = (pop_size,w,c1,c2,refreshing_gap)
    middle_params_set = CreateSets(middle_params)

    #lower
    pop_size = [48]
    w = [0.5]
    c1 = [1.2] 
    c2 = [1.0] 
    neighborhood_size =[3] 

    lower_params = (pop_size,w,c1,c2,neighborhood_size)
    lower_params_set = CreateSets(lower_params)

    num_var_set = [3]
    generation_change_set = [3]
    epochs_set = [1]

    params_sets = [upper_params_set,middle_params_set,lower_params_set,generation_change_set,
                num_var_set,epochs_set]
    
    #run tuner
    tuner = ParamTuner(PSO, functions, num_of_tests ,params_sets,range_of_params)
    tuner.Tune(verbose_level = 1,export=False,plot_gbest=(False),plot_velocity=(False))
    tuner.PrintReslt2(num_of_best=3)

if __name__ == "__main__":
    main()
