import numpy as np
import pandas as pd


class ParticleGenerator():
    
    def __init__(self,pop_size: int,dimensions: int,range_of_params: tuple,version: int):
        particles = []
        for i in range(pop_size):
            position = []
            for dim in range(dimensions):
                position.append(np.random.random()*(
                    range_of_params[1]-range_of_params[0]) + range_of_params[0]) 
            particles.append(position)
            
        df = pd.DataFrame(particles)
        df.to_csv(f"n{pop_size}_dim{dimensions}_range_{range_of_params[0]}_{range_of_params[1]}_v{version}.csv")


def main() -> None:
    n = 10
    dim = 100
    versions = 5
    ranges = [(-500.0,500.0),(-100.0,100.0),(-10.0,10.0),(-1.28,1.28),(-5.12,5.12)]

    for range1 in ranges:
        for v_id in range(versions):
            pg = ParticleGenerator(n,dim,range1,v_id+1)

if __name__ == "__main__":
    main()
