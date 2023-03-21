# PSO Authorial Variants

### Here you can find my original versions of PSO. 

## Social Classes PSO
#### The idea behind this is to create 3 populations and each one of them updates differently. The upper population is focused on exploring current global best. The middle population is focused on exploring a wider area around global best. When a lower population is focused on searching as much area as possible and exploring local optimas there. Every few iterations of the algorithm all particles are regrouped based on their fitness function values. The velocity updated in the upper and middle population is like in classical PSO but in the lower population, for each particle, there are drawn 2 other particles from the population and the particle follows the better one.

## Social Classes PSO with surrogate function

#### This algorithm works almost the same as the first one. Additionally, in each iteration, there are generated more particle based on current particles. Their fitness function values are estimated and then we choose the best particles to fill our 3 populations. For the chosen ones are calculated real fitness function values.

## Information:
### Upper population:
- inertia weight - constant
- c1 acceleration coefficient - constant (smaller than c2)
- c2 acceleration coefficient - constant (bigger than c1)
- topology - star
- using personal best position and global best position

### Middle population:
- inertia weight - constant
- c1 acceleration coefficient - constant (bigger than c2)
- c2 acceleration coefficient - constant (smaller than c1)
- topology - star
- using personal best position and global best position

### Lower population:
- inertia weight - constant
- c1 acceleration coefficient - constant (bigger than c2)
- c2 acceleration coefficient - constant (smaller than c1)
- topology - "ring"
- using personal best position and local best position


### Research papers:
 - not published yet
