Fix analyzer because it shouldn't take that long


Run Coex calculations at equilbirum at 16, 20, 40

Cut in half the initial mu range
Keep eps tight spacing

beta mu_coex vs L posotive slope

beta eps_critical vs L

# Susceptibility 

## Data cleaning
Write an algo that removes points that are 10% off from there nearest neighbour  

With Data we already have:
For 12.5x write a script that counts number of jumps J and gives <J> vs L. 

From now on only look at eps_crit +/- 0.1

# FOLDER STRUCUTURE
vd7294/ 
    _COEX_CALC
        _16_160__S1_E2.0_DF0.0_DMU0.0_K1.0
            // mu values
            _4.2
            _4.3
            _coex_analysis csv
            coex_figure.png
    _SUSC_RUNS/
        _48_48_S1_DF0.0_DMU0.0_K1.0
            _-1.76
            _-1.77



DO NOT SAVE .npy as png for Sus

ToDo:
<J> vs L for 12.5 x

get coex calculations running for 20x200 and 40x400 
beta mu_coex vs L posotive slope
beta eps_critical vs L

# Running Script

At end of sus runner have it check that for given L and eps 