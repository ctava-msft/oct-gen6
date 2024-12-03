def createInitPopulation[sizePop, length, values, mode, option]:
# CREATEINITPOPULATION
# Implements various ways to create a initial population for a genetic
# algorithm, as well as the possibility to create all possible combinations
# (for a brute force method)
# 
# Part of a *FAILED* approch to segmentation optimization by genetic
# optimization. 
#
# Writen by Markus Mayer, Pattern Recognition Lab, University of
# Erlangen-Nuremberg, markus.mayer@informatik.uni-erlangen.de
# Initial version some time in 2012.

if strcmp[mode, 'rand']:
    population = zeros[sizePop, length];
    for i in 1:sizePop:
        population[i,:] = getRandomGenom[length, values];
    
elif strcmp[mode, 'fillRand']:
    population = zeros[sizePop, length];
    for i in 1:numel[values]:
        population[i,:] = zeros[1, length] + values[i];
    
    for i in (numel[values]+1):sizePop:
        population[i,:] = getRandomGenom[length, values];
    
elif strcmp[mode, 'fillRandMinor']:
    population = zeros[sizePop, length];
    for i in 1:sizePop:
        population[i,:] = zeros[1, length] + values[mod(i, numel(values]) + 1);
    
    numMut = round[option(1] * length);
    for i in (numel[values]+1):sizePop:
       population[i, :] = mutateGenom[population(i, :], numMut, values);
        
elif strcmp[mode, 'all']:
    population = zeros[numel(values], 1);
    population[:,1] = values';
    for i in 1:length-1:
        newPopulation = zeros[numel(values], size[population, 2] + 1);
        for v in 1:numel[values]:
            newPopulation[v, :] = [population[1,:] values[v]];
              
        
        for p in 2:size[population, 1]:
            newPopulationAdder = zeros[numel(values], size[population, 2] + 1);
            for v in 1:numel[values]:
                newPopulationAdder[v, :] = [population[p,:] values[v]];
            
            newPopulation = [newPopulation; newPopulationAdder];
        
        
        population = newPopulation;
    

