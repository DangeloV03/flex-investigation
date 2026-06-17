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

Steps:
Run a long simulation that takes a couple of minitues. Request 4 CPUS and run 4 parallel instances and check that output runs correctly and outputs to the scratch folder

# Push To Git
git add ...
git commit -m "..."
git push origin dangelo/run-on-della

# Update on Della
cd /scratch/gpfs/WJACOBS/vd7294/flex-investigation
git status
git pull origin dangelo/run-on-delta  

# Reset daemon processes
# find and kill old processes, then:
nohup python run_all.py > run_all.log 2>&1 &
nohup python analyzer.py > analyzer.log 2>&1 &