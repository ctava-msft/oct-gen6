def killPopulationDuplicates[population]:
# KILLPOPULATIONDUPLICATES
# Kills the duplicates in a population
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

newPopulation = population[1,:];
while numel[population] != 0:
    if size[population, 1] >= 2:
        population = population[2:, :];
    else:
        break;
    
    
    keep = zeros[size(population, 1], 1) + 1;
    for i in 1:size[population, 1]:
        if all[newPopulation(,:] == population[i,:]):
            keep[i] = 0;
        
    
    
    population = population[keep == 1, :];
    if size[population, 1] > 1:
        newPopulation = [newPopulation; population[1,:]];
    
