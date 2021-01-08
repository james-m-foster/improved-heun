import math
import numpy as np

'''
Numerical experiment for the Euler-Maruyama, Heun and improved Heun methods
on the additive noise SDE:

dy_t = sin(y_t) dt + sigma dW_t

'''

# parameters
sigma = 1.0 
y0 = 1.0

# vector field
def f(y):         
    return math.sin(y)

fy0 = f(y0)

root_six_over_three = math.sqrt(6)/3
root_six_over_three_sigma = root_six_over_three*sigma
half_minus_root_six_over_six = 0.5 - (math.sqrt(6))/6
half_plus_root_six_over_six = 0.5 + (math.sqrt(6))/6

# numerical methods
def EulerMaruyama(y, h, w):
    return y + f(y)*h + sigma*w

def Heun(y, h, w):
    fy = f(y)
    y1 = y + fy*h + sigma*w

    return y + 0.5*(fy + f(y1))*h + sigma*w

def AdHoc(y, fy, h, w):
    y1 = y + fy * h + sigma * w
    fy1 = f(y1)
    y2 = y + 0.5*(fy + fy1) * h + sigma*w
    
    return y2, fy1

def ASH(y, h, w, area):
    y1 = y + sigma * (half_minus_root_six_over_six * w + area)
    
    fy1 = f(y1)
    
    y2 = y1 + fy1 * h + root_six_over_three_sigma*w
    
    return y + 0.5*(fy1 + f(y2)) * h + sigma * w

def SRK(y, h, w, area):
    y1 = y + sigma * (half_plus_root_six_over_six * w + area)
        
    y2 = y + f(y) * h + sigma * (half_minus_root_six_over_six * w + area)
    
    return y + 0.5*(f(y1) + f(y2)) * h + sigma * w

# interval length
T = 1.0

# no of simulations
simulations = 1000000

# no of steps used by the fine and coarse approximations
no_of_crude_steps = 3
no_of_fine_steps = 10

# step sizes
crude_stepsize = T/no_of_crude_steps
fine_stepsize = T/(no_of_crude_steps*no_of_fine_steps)
one_over_step_size = 1 / crude_stepsize
sqrt_fine_stepsize = math.sqrt(fine_stepsize)
root_twelve_sqrt_fine_stepsize = math.sqrt(fine_stepsize)/math.sqrt(12)

strong_em_error = 0
mean_em_error =  0
sec_moment_em_error =  0

strong_h_error = 0
mean_h_error =  0
sec_moment_h_error =  0

strong_ah_error = 0
mean_ah_error =  0
sec_moment_ah_error =  0

strong_ash_error = 0
mean_ash_error = 0
sec_moment_ash_error =  0

strong_srk_error = 0
mean_srk_error = 0
sec_moment_srk_error =  0

#initial conditions
crude_em_y = y0

fine_h_y = y0
crude_h_y = y0

crude_ah_y = y0
crude_ah_fy = fy0

fine_ash_y = y0
crude_ash_y = y0

fine_srk_y = y0
crude_srk_y = y0

fine_w = 0
crude_w = 0

fine_area = 0
crude_area = 0

# Monte carlo simulations for increment only methods
for k in range(simulations):   
    for i in range(no_of_crude_steps):
        # reset the coarse increment of Brownian motion
        crude_w = 0
        for j in range(no_of_fine_steps):
            # generate brownian increment on the fine scale
            fine_w = np.random.normal(0.0, sqrt_fine_stepsize)
            
            # compute an approximation on the fine scale
            fine_h_y = Heun(fine_h_y, fine_stepsize, fine_w)
            
            # combine fine brownian increments to the coarse scale
            crude_w = crude_w + fine_w
        
        # compute approximations on the coarse scale
        crude_em_y = EulerMaruyama(crude_em_y, crude_stepsize, crude_w)
                    
        crude_h_y = Heun(crude_h_y, crude_stepsize, crude_w)

        crude_ah_y, crude_ah_fy = AdHoc(crude_ah_y, crude_ah_fy , crude_stepsize, crude_w)
    
    # sum the weak and strong errors from each sample
    strong_em_error = strong_em_error + (crude_em_y - fine_h_y)**2
    mean_em_error = mean_em_error + crude_em_y - fine_h_y
    sec_moment_em_error = sec_moment_em_error + crude_em_y**2 - fine_h_y**2

    strong_h_error = strong_h_error + (crude_h_y - fine_h_y)**2
    mean_h_error = mean_h_error + crude_h_y - fine_h_y
    sec_moment_h_error = sec_moment_h_error + crude_h_y**2 - fine_h_y**2
          
    strong_ah_error = strong_ah_error + (crude_ah_y - fine_h_y)**2
    mean_ah_error = mean_ah_error + crude_ah_y - fine_h_y
    sec_moment_ah_error = sec_moment_ah_error + crude_ah_y**2 - fine_h_y**2
    
    
    # revert to initial conditions 
    crude_em_y = y0

    fine_h_y = y0
    crude_h_y = y0

    crude_ah_y = y0
    crude_ah_fy = fy0
    
# compute the Monte carlo error estimators
strong_em_error = strong_em_error /  simulations
mean_em_error = mean_em_error /  simulations
sec_moment_em_error = sec_moment_em_error /  simulations

strong_h_error = strong_h_error /  simulations
mean_h_error = mean_h_error /  simulations
sec_moment_h_error = sec_moment_h_error /  simulations
    
strong_ah_error = strong_ah_error /  simulations
mean_ah_error = mean_ah_error /  simulations
sec_moment_ah_error = sec_moment_ah_error /  simulations

# display results
print('Strong error for Euler-Maruyama: ', math.sqrt(strong_em_error))
print('Mean error for Euler-Maruyama: ', abs(mean_em_error))
print('Second moment error for Euler-Maruyama: ', abs(sec_moment_em_error), '\n')
print('Strong error for Heun: ', math.sqrt(strong_h_error))
print('Mean error for Heun: ', abs(mean_h_error))
print('Second moment error for Heun: ', abs(sec_moment_h_error), '\n')
print('Strong error for AdHoc: ', math.sqrt(strong_ah_error))
print('Mean error for AdHoc: ', abs(mean_ah_error))
print('Second moment error for AdHoc: ', abs(sec_moment_ah_error), '\n')


# Monte carlo simulations for "increment + area" methods
for k in range(simulations):   
    for i in range(no_of_crude_steps):
        # reset the coarse increment of Brownian motion
        crude_w = 0
        crude_area = 0
        
        for j in range(no_of_fine_steps):
            # generate brownian increment on the fine scale
            fine_w = np.random.normal(0.0, sqrt_fine_stepsize)
            fine_area = np.random.normal(0.0, root_twelve_sqrt_fine_stepsize)
            
            # compute an approximation on the fine scale
            fine_ash_y = ASH(fine_ash_y, fine_stepsize, fine_w, fine_area)
            fine_srk_y = ASH(fine_srk_y, fine_stepsize, fine_w, fine_area)
            
            # combine fine brownian increments/areas to the coarse scale           
            crude_area = crude_area + fine_stepsize * (crude_w + 0.5*fine_w + fine_area);
                                     
            crude_w = crude_w + fine_w
            
        crude_area = crude_area*one_over_step_size - 0.5*crude_w;
        
        # compute approximations on the coarse scale                   
        crude_ash_y = ASH(crude_ash_y, crude_stepsize, crude_w, crude_area)
        crude_srk_y = SRK(crude_srk_y, crude_stepsize, crude_w, crude_area)
        
    # sum the weak and strong errors from each sample
    strong_ash_error = strong_ash_error + (crude_ash_y - fine_ash_y)**2
    mean_ash_error = mean_ash_error + crude_ash_y - fine_ash_y
    sec_moment_ash_error = sec_moment_ash_error + crude_ash_y**2 - fine_ash_y**2

    strong_srk_error = strong_srk_error + (crude_srk_y - fine_srk_y)**2
    mean_srk_error = mean_srk_error + crude_srk_y - fine_srk_y
    sec_moment_srk_error = sec_moment_srk_error + crude_srk_y**2 - fine_srk_y**2


    # revert to initial conditions 
    fine_ash_y = y0
    crude_ash_y = y0

    fine_srk_y = y0
    crude_srk_y = y0 
    
    
# compute the Monte carlo error estimators
strong_ash_error = strong_ash_error / simulations
mean_ash_error = mean_ash_error / simulations
sec_moment_ash_error = sec_moment_ash_error / simulations

strong_srk_error = strong_srk_error / simulations
mean_srk_error = mean_srk_error / simulations
sec_moment_srk_error = sec_moment_srk_error / simulations

# display results
print('Strong error for ASH: ', math.sqrt(strong_ash_error))
print('Mean error for ASH: ', abs(mean_ash_error))
print('Second moment error for ASH: ', abs(sec_moment_ash_error))
print('Strong error for SRK: ', math.sqrt(strong_srk_error))
print('Mean error for SRK: ', abs(mean_srk_error))
print('Second moment error for SRK: ', abs(sec_moment_srk_error))