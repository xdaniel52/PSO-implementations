import xml.etree.ElementTree as gfg 
import sys
sys.path.append('.')    
from PSO_authorial_variants.social_classes_pso import PSO as PSOparent
from test_functions import * 

class PSO(PSOparent):
    def __init__(self, upper_pop_size ,upper_w, upper_c1, upper_c2,\
                 middle_pop_size, middle_w, middle_c1, middle_c2,\
                 lower_pop_size, lower_w, lower_c1, lower_c2, lower_neighborhood_size,\
                 generation_change, num_var,epochs,
                 ) -> None:
        
        upper_params = (upper_pop_size,upper_w,upper_c1,upper_c2)
        middle_params = (middle_pop_size,middle_w,middle_c1,middle_c2)
        lower_params = (lower_pop_size,lower_w,lower_c1,lower_c2,lower_neighborhood_size)

        super().__init__(upper_params,middle_params,lower_params,generation_change,num_var,epochs)
         
class TunerPSO:
    """Searching for best params for inner PSO using outer PSO"""   
    def __init__(self, outer_pso_class, outer_params: list, inner_pso_class, pso_range_of_params: list, num_of_repetitions: int) -> None:
        self.PSO = outer_pso_class(*outer_params)
        self.inner_pso_class = inner_pso_class
        self.pso_range_of_params = pso_range_of_params
        self.test_num = num_of_repetitions 
        self.history_params = []
        self.history_best = []
        
    def Tune(self, fun, range_of_params: list) -> None:
        new_fun = self.Create_New_Function(fun, range_of_params,self.pso_range_of_params)
        self.PSO.Start(new_fun,self.pso_range_of_params,False,True)
        self.PSO.Print_result()
    
    def Create_New_Function(self, fun, range_of_params: list,pso_range_of_params: list):
        def New_Fun(*args):
            for i in range(len(args[0])):
                if i in [0, 4, 8, 12, 13, 14, 15, 16, 19]:  # 1.0 precision params
                    args[0][i] = round(args[0][i])
                    
                    if(args[0][i] < pso_range_of_params[i][0]):
                        args[0][i] = pso_range_of_params[i][0]
                    elif(args[0][i] > pso_range_of_params[i][1]):
                      args[0][i] = pso_range_of_params[i][1]
            
                elif i == 17: # 0.1 precision params
                    args[0][i] = round(args[0][i],-1)
                    
                    if(args[0][i] < pso_range_of_params[i][0]):
                        args[0][i] = pso_range_of_params[i][0]
                    elif(args[0][i] > pso_range_of_params[i][1]):
                      args[0][i] = pso_range_of_params[i][1]
                elif i == 18: # 0.01 precision params
                    args[0][i] = round(args[0][i],-2)
                    
                    if(args[0][i] < pso_range_of_params[i][0]):
                        args[0][i] = pso_range_of_params[i][0]
                    elif(args[0][i] > pso_range_of_params[i][1]):
                      args[0][i] = pso_range_of_params[i][1]
                    
                else: # 10.0 precision params
                    args[0][i] = round(args[0][i],1)
                    
                
            if args[0] in self.history_params:
                return self.history_best[self.history_params.index(args[0])]
            else:              
                pso = self.inner_pso_class(*args[0])
                results = []
                for i in range(self.test_num):
                    pso.Start(fun, range_of_params)
                    results.append(pso.GetOptimum())
                results.sort(key = lambda x: x)

                best_value = results[len(results)//2]
                self.history_params.append(args[0].copy())
                self.history_best.append(best_value)
                
                return best_value
            
        return New_Fun
        
    def AddElementToTree(self, parent: gfg.Element, elem_name: str, text: str) -> None:
        elementXML = gfg.Element(elem_name) 
        parent.append(elementXML)
        elementXML.text = str(text)
        
    
    def Create_XML_Tree(self, fun_Name: str, params: list,global_best: float) -> None:
        testXML = gfg.Element('TEST') 
        
        self.AddElementToTree(parent = testXML,elem_name = 'PSO_NAME',text = "Social Classes PSO")
        self.AddElementToTree(parent = testXML,elem_name = 'FUNCTION',text = fun_Name)
        
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_POP_SIZE',text = params[0])
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_W',text = params[1])
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_C1',text = params[2])
        self.AddElementToTree(parent = testXML,elem_name = 'UPPER_C2',text = params[3])

        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_POP_SIZE',text = params[4])
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_W',text = params[5])
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_C1',text = params[6])
        self.AddElementToTree(parent = testXML,elem_name = 'MIDDLE_C2',text = params[7])
        
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_POP_SIZE',text = params[8])
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_W',text = params[9])
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_C1',text = params[10])
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_C2',text = params[11])
        self.AddElementToTree(parent = testXML,elem_name = 'LOWER_NEIGHBORHOOD_SIZE',text = params[12])
        
        
        self.AddElementToTree(parent = testXML,elem_name = 'GENERATION_CHANGE',text = params[13])
        self.AddElementToTree(parent = testXML,elem_name = 'NUM_VAR',text = params[14])    
        self.AddElementToTree(parent = testXML,elem_name = 'EPOCHS',text = params[15])
        
        self.AddElementToTree(parent = testXML,elem_name = 'SIGMA_TOP_PAR',text = params[16])
        self.AddElementToTree(parent = testXML,elem_name = 'SIGMA_BOT_PAR',text = params[17])    
        self.AddElementToTree(parent = testXML,elem_name = 'PROB_BASE',text = params[18])
        self.AddElementToTree(parent = testXML,elem_name = 'PROB_SCALE',text = params[19])
        
        self.AddElementToTree(parent = testXML,elem_name = 'GLOBAL_BEST',text = global_best)
        
        return testXML
   
    def ExportResultToXML(self, file_Name: str, fun_Name: str) -> None:
        root = gfg.Element('ROOT')
        testsXML = gfg.Element('TESTS')
        root.append(testsXML)
        
        for i in range(len(self.history_params)):       
            testXML = self.Create_XML_Tree(fun_Name,self.history_params[i],self.history_best[i])
            testsXML.append(testXML)
            
        tree = gfg.ElementTree(root)     
        with open (file_Name+".xml","wb") as file :
            tree.write(file)                            
        

def main() -> None:
    #list of all functions
    functions = [f_1 , f_2 , f_3 , f_4 , f_5 , f_6 , f_7 , f_8 , f_9, ]
    range_of_params = [(-100.,100.),(-10.,10.),(-10.,10.),(-10.,10.),(-100.,100.),(-1.28,1.28),(-500.,500.),(-5.12,5.12),(-5.12,5.12) ]

    #choose functions for test
    functions = functions[:]
    range_of_params = range_of_params[:]
    assert len(functions) == len(range_of_params)

    #define parameters for outer pso
    outer_params =  [5,0.2,0.5,1.5,
                    5,0.4,1.0,1.3,
                    20,0.5,1.2,1.0,5,
                    4,
                    20,
                    10,
                    5,10,0.1,0.30  ]

    # 0 4 8 12 13 14 15 16 17 ids of static params, not for optimization

    num_of_repetitions = 1

    #define search space for params 
    range_upper_pop_size = ( 5, 5)
    range_upper_w = ( 0.1, 0.7)
    range_upper_c1 = ( 0.1, 2.5)
    range_upper_c2 = ( 0.1, 2.5)
    range_middle_pop_size = ( 5, 5)
    range_middle_w = ( 0.1, 0.9)
    range_middle_c1 = ( 0.1, 2.5)
    range_middle_c2 = ( 0.1, 2.5)
    range_lower_pop_size = ( 30, 30)
    range_lower_w = ( 0.4, 0.9)
    range_lower_c1 = ( 0.5, 2.5)
    range_lower_c2 = ( 0.5, 2.5)
    range_lower_neighborhood_size = ( 5, 5)
    range_generation_change = ( 0, 10)
    range_num_var = ( 2, 2)
    range_epochs = ( 100, 100)
    range_sigma_top_par = ( 10, 1000)
    range_sigma_bot_par = ( 100, 1000000)
    range_prob_base = ( 0, 5)
    range_prob_scale = ( 5, 30)
                                  
    pso_range_of_params = [range_upper_pop_size,range_upper_w,range_upper_c1,range_upper_c2,
                        range_middle_pop_size,range_middle_w,range_middle_c1,range_middle_c2,
                        range_lower_pop_size,range_lower_w,range_lower_c1,range_lower_c2, range_lower_neighborhood_size,
                        range_generation_change, range_num_var, range_epochs,
                        range_sigma_top_par,range_sigma_bot_par,range_prob_base,range_prob_scale
                        ]

    #run tuner
    tuner = TunerPSO(PSO, outer_params, PSO, pso_range_of_params, num_of_repetitions)
    tuner.Tune(functions[1],[range_of_params[1] for i in range(range_num_var[1])])
    tuner.ExportResultToXML("test_result_1234 ","modified_schwefel")
    
if __name__ == "__main__":
    main()


