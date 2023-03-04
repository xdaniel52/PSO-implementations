import numpy as np
import sys
sys.path.append('.')
from PSO.pso import PSO
from test_functions import *                

class TestPSOVariants:
    
    def __init__(self, PSO_list: list, functions_list: list, range_of_params_list: list,num_of_repetitions: int):
        self.PSO_list = PSO_list
        self.functions_list = functions_list
        self.range_of_params_list = range_of_params_list
        self.num_of_repetitions = num_of_repetitions
        self.each_result =[[[0 for i in range(num_of_repetitions)] for j in functions_list] for k in range(len(PSO_list))]

    def Run(self, verbose: bool = False):
        for pso_id in range(len(self.PSO_list)):            
            for fun_id in range(len(self.functions_list)):               
                for test_id in range(self.num_of_repetitions):
                    self.PSO_list[pso_id].Start(self.functions_list[fun_id], [self.range_of_params_list[fun_id] for _ in range(self.PSO_list[pso_id].num_var)])
                    self.each_result[pso_id][fun_id][test_id] = self.PSO_list[pso_id].GetOptimum()
                    
    def PrintReslt(self):
        print(f"Raport for:")
        print(f"PSO_list: {[pso.__class__.__name__ for pso in self.PSO_list]}")
        print(f"functions_list: {[fun.__name__ for fun in self.functions_list]}")
        print(f"num_of_tests: {self.num_of_repetitions}")
        print(f"Results: ")
        for pso_id in range(len(self.PSO_list)):  
            print(f"{pso_id} {self.PSO_list[pso_id].__class__.__name__}")
            for fun_id in range(len(self.functions_list)):
                avg = sum(self.each_result[pso_id][fun_id])/self.num_of_repetitions
                median = np.median(self.each_result[pso_id][fun_id])
                std = np.std(self.each_result[pso_id][fun_id])
                maks = max(self.each_result[pso_id][fun_id])
                mini = min(self.each_result[pso_id][fun_id])

                print(f"\t{fun_id} {self.functions_list[fun_id].__name__}")
                print(f"\t\tAvg: {avg}\tMedian: {median}\tStd: {std}\t{maks}Maks: \tMin: {mini}".replace('.',','))
            print()
                                    
    def PrintDetailedResult(self):
        print(f"Raport for:")
        print(f"PSO_list: {[pso.__class__.__name__ for pso in self.PSO_list]}")
        print(f"functions_list: {[fun.__name__ for fun in self.functions_list]}")
        print(f"num_of_tests: {self.num_of_repetitions}")
        print(f"Results: ")
        for pso_id in range(len(self.PSO_list)):  
            print(pso_id+": ")
            for fun_id in range(len(self.functions_list)):
                print("      "+fun_id+": ")
                for test_id in range(self.num_of_repetitions):
                    print("            "+fun_id+": " + self.each_result[pso_id][fun_id][test_id])
    

def main() -> None:
    #list of all functions
    functions = [f_1 , f_2 , f_3 , f_4 , f_5 , f_6 , f_7 , f_8 , f_9, ]
    range_of_params = [(-100.,100.),(-10.,10.),(-10.,10.),(-10.,10.),(-100.,100.),(-1.28,1.28),(-500.,500.),(-5.12,5.12),(-5.12,5.12) ]

    #choose functions for test
    functions = functions[:]
    range_of_params = range_of_params[:]
    assert len(functions) == len(range_of_params)

    pso_list = []

    #define parameters of tested PSO 1
    num_var = 5
    pop_size = 30
    w = 0.7
    c1 = 1.49
    c2 = 1.49
    epochs = 100
    pso_list.append(PSO(num_var,pop_size,w,c1,c2,epochs))

    #define parameters of tested PSO 2
    num_var = 5
    pop_size = 30
    w = 0.7
    c1 = 1.49
    c2 = 1.49
    epochs = 200

    pso_list.append(PSO(num_var,pop_size,w,c1,c2,epochs))

    #set number of repetitions for each function
    num_of_repetitions = 3

    test_variants = TestPSOVariants(pso_list,functions, range_of_params, num_of_repetitions)
    test_variants.Run()
    test_variants.PrintReslt()
    
if __name__ == "__main__":
    main()



