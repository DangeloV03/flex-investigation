Goal:

Create a data generation pipeline that gets
epsilon delta mu Scheme, delta f and L?
outputs beta mu_coex FLEX and beta mu_coex SIM

epsilon bounds = 

Gen Samples:
Creates massive data set of all combinations of epsilon, delta mu, and L

JSON runner:
takes in JSON as params
outputs average density of active and empty as function of input params

Analyzer:
Takes in output of JSON runner
and calculates beta mu_coex Sim by averaing simulations

Analyzer (check):
First check with sigmoid function 
if that flips from neg to posotive then we plot 
abs(density_active - density_empty) and find minimum 

Submit a Slurm request

Note (Initial Condition): 0 to 0.5 Lx is active and rest is empty



FUNCTIONAL DECOMPOSITION

1. for open hetero chain mu VT file
    - make it callable from python
    - TEST: write python file that simply runs it
2. Write JSON runner that takes as input delta mu epsilon delta F and Lx and Ly and Initial Condition (half filled half empty), Scheme, num_parallel_runs, eq_time, prod_time and outputs density of empty active and inactive (STILL DO CHUNK THINK)

3. In JSON runner have it run n times in parallel. set n to low number while running on laptop

4. In JSON runner output 
- final_lattice_i.npy
- density as time series over short production run (1/10 of eq) 
- time average density of active, inert, empty 
- output csv
* id, <rho_active>, <rho_inert>, <rho_empty>, time

NOTE: ONLY LOGIC IN JSON RUNNER: If output.csv is already there it should append and adjust ID's so that they are sequential. 



Quick Fixes: 
1. Switch over to Conda
2. Check the lattice-gas repo and try to implement periodic bondary conditions for vertical but not horizontal

3. FOR ME: read the SR paper