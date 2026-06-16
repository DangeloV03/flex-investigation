ssh vd7294@della.princeton.edu

[vd7294@della9 ~]$ cd /scratch/gpfs/WJACOBS



Change:
Have run all always run and checks how many slurm files there are and submits the first 100. Periodiclly checks how many jobs are active and submits more until it clears the sample directory. 

Analyzer will then append its job to the run all folder so if you have to go back and ask for more data it will just be put back in run all 

Analyzer will be dorment as it waits and ramps up when it sees new data available 


In analyzer append it to the folders that get run as a stack so it is first


Install Cargo
Install lattice-gas module
Set up conda env 
Write smokescreen python test that we run via slurm to test that it is properly being called