import pandas as pd

class ImportParticles:
    def __init__(self, num_of_repetitions, size, dimensions, rp):
        self.import_version = 0
        self.num_of_repetitions = num_of_repetitions
        self.size = size
        self.dimensions = dimensions
        self.rp = rp
        
    def Import(self) -> list:
        particles = []
        self.import_version+=1
        self.file_name = f"particle_sets/n{self.size}_dim{self.dimensions}_range_{self.rp[0]}_{self.rp[1]}_v{self.import_version}.csv"   
        self.import_version = self.import_version%self.num_of_repetitions

        imported = pd.read_csv(self.file_name,index_col=0)
        for p in range(self.size):     
            particles.append(list(imported.loc[p].to_dict().values()))

        return particles
    


