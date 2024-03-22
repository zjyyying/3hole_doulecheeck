import subprocess
import os
import sys
import time
import shutil
import math
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

import parameters as pam
import hamiltonian as ham
import lattice as lat
import variational_space as vs 
import utility as util
import lanczos

def reorder_z(slabel):
    '''
    reorder orbs such that d orb is always before p orb and Ni layer (z=1) before Cu layer (z=0)
    '''
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    
    state_label = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    
    if orb1 in pam.Ni_Cu_orbs and orb2 in pam.Ni_Cu_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        elif z2==z1 and orb1=='dx2y2' and orb2=='d3z2r2':
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]          
           
    elif orb1 in pam.O_orbs and orb2 in pam.Ni_Cu_orbs:
        state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
        
    elif orb1 in pam.O_orbs and orb2 in pam.O_orbs:
        if z2>z1:
            state_label = [s2,orb2,x2,y2,z2,s1,orb1,x1,y1,z1]
            
    return state_label
                
def make_z_canonical(slabel):
    
    s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
    s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
    s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14];  
    '''
    For three holes, the original candidate state is c_1*c_2*c_3|vac>
    To generate the canonical_state:
    1. reorder c_1*c_2 if needed to have a tmp12;
    2. reorder tmp12's 2nd hole part and c_3 to have a tmp23;
    3. reorder tmp12's 1st hole part and tmp23's 1st hole part
    '''
    tlabel = [s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2]
    tmp12 = reorder_z(tlabel)

    tlabel = tmp12[5:10]+[s3,orb3,x3,y3,z3]
    tmp23 = reorder_z(tlabel)

    tlabel = tmp12[0:5]+tmp23[0:5]
    tmp = reorder_z(tlabel)

    slabel = tmp+tmp23[5:10]
   
                
    return slabel


def get_ground_state(matrix, VS, S_Ni_val, Sz_Ni_val, S_Cu_val, Sz_Cu_val):  
    '''
    Obtain the ground state info, namely the lowest peak in Aw_dd's component
    in particular how much weight of various d8 channels: a1^2, b1^2, b2^2, e^2
    '''        
    t1 = time.time()
    print ('start getting ground state')
#     # in case eigsh does not work but matrix is actually small, e.g. Mc=1 (CuO4)
#     M_dense = matrix.todense()
#     print ('H=')
#     print (M_dense)
    
#     for ii in range(0,1325):
#         for jj in range(0,1325):
#             if M_dense[ii,jj]>0 and ii!=jj:
#                 print ii,jj,M_dense[ii,jj]
#             if M_dense[ii,jj]==0 and ii==jj:
#                 print ii,jj,M_dense[ii,jj]
                    
                
#     vals, vecs = np.linalg.eigh(M_dense)
#     vals.sort()                                                               #calculate atom limit
#     print ('lowest eigenvalue of H from np.linalg.eigh = ')
#     print (vals)
    
    # in case eigsh works:
    Neval = pam.Neval
    vals, vecs = sps.linalg.eigsh(matrix, k=Neval, which='SA')
    vals.sort()
    print ('lowest eigenvalue of H from np.linalg.eigsh = ')
    print (vals)
    print (vals[0])
    
    if abs(vals[0]-vals[3])<10**(-5):
        number = 4
    elif abs(vals[0]-vals[2])<10**(-5):
        number = 3        
    elif abs(vals[0]-vals[1])<10**(-5):
        number = 2
    else:
        number = 1
    print ('Degeneracy of ground state is ' ,number)        

    wgt_d9d8 = np.zeros(6)
    wgt_d8d9 = np.zeros(6)
    wgt_d9d9L = np.zeros(6)        
    wgt_d9d10L2= np.zeros(6)
    wgt_d10d9L2= np.zeros(6)
    wgt_d9L2d10= np.zeros(6)   
    wgt_d10Ld9L= np.zeros(6)  
    wgt_d9Ld10L= np.zeros(6)
    wgt_d10L2d9= np.zeros(6)        
    wgt_d10d8L= np.zeros(6)
    wgt_d9Ld9 = np.zeros(6)
    wgt_d8d10L = np.zeros(6)        
    wgt_d8Ld10 = np.zeros(6) 
    wgt_d10Ld8 = np.zeros(6)  
    wgt_d10d10 = np.zeros(6)         

        
    
    
    #get state components in GS and another 9 higher states; note that indices is a tuple
    for k in range(0,number):                                                                          #gai
        #if vals[k]<pam.w_start or vals[k]>pam.w_stop:
        #if vals[k]<11.5 or vals[k]>14.5:
        #if k<Neval:
        #    continue
            
        print ('eigenvalue = ', vals[k])
        indices = np.nonzero(abs(vecs[:,k])>0.1)



#         s11=0
#         s10=0        
#         s01=0
#         s00=0        
        #Sumweight refers to the general weight.Sumweight1 refers to the weight in indices.Sumweight_picture refers to the weight that is calculated.Sumweight2 refers to the weight that differs by orbits

        sumweight=0
        sumweight1=0
        synweight2=0
        # stores all weights for sorting later
        dim = len(vecs[:,k])
        allwgts = np.zeros(dim)
        allwgts = abs(vecs[:,k])**2
        ilead = np.argsort(-allwgts)   # argsort returns small value first by default
            

        total = 0

        print ("Compute the weights in GS (lowest Aw peak)")
        
        #for i in indices[0]:
        for i in range(dim):
            # state is original state but its orbital info remains after basis change
            istate = ilead[i]
            weight = allwgts[istate]
            
            #if weight>0.01:

            total += weight
                
            state = VS.get_state(VS.lookup_tbl[istate])

        
            s1 = state['hole1_spin']
            s2 = state['hole2_spin']
            s3 = state['hole3_spin']          
            orb1 = state['hole1_orb']
            orb2 = state['hole2_orb']
            orb3 = state['hole3_orb']         
            x1, y1, z1 = state['hole1_coord']
            x2, y2, z2 = state['hole2_coord']
            x3, y3, z3 = state['hole3_coord']        

            #if abs(x1)>1. or abs(y1)>1. or abs(x2)>1. or abs(y2)>1.:
            #    continue
            S_Ni_12  = S_Ni_val[istate]
            Sz_Ni_12 = Sz_Ni_val[istate]
            S_Cu_12  = S_Cu_val[istate]
            Sz_Cu_12 = Sz_Cu_val[istate]
            
#             S_Niother_12  = S_other_Ni_val[i]
#             Sz_Niother_12 = Sz_other_Ni_val[i]
#             S_Cuother_12  = S_other_Cu_val[i]
#             Sz_Cuother_12 = Sz_other_Cu_val[i]
             
#             print (' state ', i, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',orb4,s4,x4,y4,z4,\
#                '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
#                ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
#                ", weight = ", weight,'\n')   
                    
    
    

            slabel=[s1,orb1,x1,y1,z1,s2,orb2,x2,y2,z2,s3,orb3,x3,y3,z3]
            slabel= make_z_canonical(slabel)
            s1 = slabel[0]; orb1 = slabel[1]; x1 = slabel[2]; y1 = slabel[3]; z1 = slabel[4];
            s2 = slabel[5]; orb2 = slabel[6]; x2 = slabel[7]; y2 = slabel[8]; z2 = slabel[9];
            s3 = slabel[10]; orb3 = slabel[11]; x3 = slabel[12]; y3 = slabel[13]; z3 = slabel[14]
 
            
            if weight > 0.01:
                sumweight1=sumweight1+abs(vecs[i,k])**2
                print (' state ', istate, ' ',orb1,s1,x1,y1,z1,' ',orb2,s2,x2,y2,z2,' ',orb3,s3,x3,y3,z3,' ',\
                   '\n S_Ni=', S_Ni_12, ',  Sz_Ni=', Sz_Ni_12, \
                   ',  S_Cu=', S_Cu_12, ',  Sz_Cu=', Sz_Cu_12, \
                   ", weight = ", weight,'\n')   

            
            if (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and z1==1 and z2==z3==0:
                wgt_d9d8[0]+=abs(vecs[istate,k])**2
                if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and S_Cu_12==0 :
                     wgt_d9d8[1]+=abs(vecs[istate,k])**2
                if orb1=='dx2y2' and  orb2=='d3z2r2'  and  orb3=='dx2y2' and  S_Cu_12==1:
                     wgt_d9d8[2]+=abs(vecs[istate,k])**2   
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and  S_Cu_12==0:
                     wgt_d9d8[3]+=abs(vecs[istate,k])**2                      
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.Ni_Cu_orbs) and z1==z2==1 and z3==0:
                wgt_d8d9[0]+=abs(vecs[istate,k])**2                    
                if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and S_Ni_12==0:
                     wgt_d8d9[1]+=abs(vecs[istate,k])**2
                if orb1=='dx2y2' and  orb2=='dx2y2'  and  orb3=='d3z2r2' and S_Ni_12==0:
                     wgt_d8d9[2]+=abs(vecs[istate,k])**2    
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  orb3=='dx2y2' and S_Ni_12==1:
                     wgt_d8d9[3]+=abs(vecs[istate,k])**2                      
               
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==1 and z2==z3==0:
                wgt_d9d9L[0]+=abs(vecs[istate,k])**2               
                if orb1=='dx2y2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')   :
                     wgt_d9d9L[1]+=abs(vecs[istate,k])**2   
                if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py') :
                     wgt_d9d9L[2]+=abs(vecs[istate,k])**2               
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py')  :
                     wgt_d9d9L[3]+=abs(vecs[istate,k])**2                       
             
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z1==1 and z2==z3==0:
                wgt_d9d10L2[0]+=abs(vecs[istate,k])**2             
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py')):
                     wgt_d9d10L2[1]+=abs(vecs[istate,k])**2   
                if orb1=='d3z2r2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py')):
                     wgt_d9d10L2[2]+=abs(vecs[istate,k])**2                       

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z1==z2==z3==0 :
                wgt_d10d9L2[0]+=abs(vecs[istate,k])**2                          
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py')):
                     wgt_d10d9L2[1]+=abs(vecs[istate,k])**2                 
                if orb1=='d3z2r2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py')):
                     wgt_d10d9L2[2]+=abs(vecs[istate,k])**2                 

            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z1==z2==z3==1 :
                wgt_d9L2d10[0]+=abs(vecs[istate,k])**2                            
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d9L2d10[1]+=abs(vecs[istate,k])**2                                    
                if orb1=='d3z2r2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')) :
                     wgt_d9L2d10[2]+=abs(vecs[istate,k])**2                     
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z2==1 and z1==z3==0 :
                wgt_d10Ld9L[0]+=abs(vecs[istate,k])**2                                       
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d10Ld9L[1]+=abs(vecs[istate,k])**2                 
                if orb1=='d3z2r2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d10Ld9L[2]+=abs(vecs[istate,k])**2                    
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z3==0 and z1==z2==2:
                wgt_d9Ld10L[0]+=abs(vecs[istate,k])**2                                       
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d9Ld10L[1]+=abs(vecs[istate,k])**2                  
                if orb1=='d3z2r2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d9Ld10L[2]+=abs(vecs[istate,k])**2                     
               
            
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.O_orbs) and (orb3 in pam.O_orbs) and z1==0 and z2==z3==1 :
                wgt_d10L2d9[0]+=abs(vecs[istate,k])**2                
                if orb1=='dx2y2' and  ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d10L2d9[1]+=abs(vecs[istate,k])**2                                            
                if orb1=='d3z2r2' and ((orb2=='px' or orb2=='py')  and  (orb3=='py' or orb3=='px')):
                     wgt_d10L2d9[4]+=abs(vecs[istate,k])**2 
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==z2==z3==0:
                wgt_d10d8L[0]+=abs(vecs[istate,k])**2                      
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Cu_12==1 :
                     wgt_d10d8L[1]+=abs(vecs[istate,k])**2       
                if  orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Cu_12==0 :
                     wgt_d10d8L[2]+=abs(vecs[istate,k])**2                       
                    
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==z3==1 and z2==0 :
                wgt_d9Ld9[0]+=abs(vecs[istate,k])**2                    
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py'):
                     wgt_d9Ld9[1]+=abs(vecs[istate,k])**2 
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py'):
                     wgt_d9Ld9[2]+=abs(vecs[istate,k])**2                     


            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==z2==1 and z3==0:
                wgt_d8d10L[0]+=abs(vecs[istate,k])**2                           
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Ni_12==1:
                     wgt_d8d10L[1]+=abs(vecs[istate,k])**2  
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Ni_12==0:
                     wgt_d8d10L[2]+=abs(vecs[istate,k])**2                      
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==z2==z3==1:
                wgt_d8Ld10[0]+=abs(vecs[istate,k])**2                     
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Ni_12==1:
                     wgt_d8Ld10[1]+=abs(vecs[istate,k])**2 
                if  orb1=='dx2y2' and  orb2=='d3z2r2' and  (orb3=='px' or orb3=='py') and S_Ni_12==0:
                     wgt_d8Ld10[2]+=abs(vecs[istate,k])**2                     
                    
            elif (orb1 in pam.Ni_Cu_orbs) and (orb2 in pam.Ni_Cu_orbs) and (orb3 in pam.O_orbs) and z1==z2==0 and z3==1:
                wgt_d10Ld8[0]+=abs(vecs[istate,k])**2                     
                if orb1=='d3z2r2' and  orb2=='dx2y2'  and  (orb3=='px' or orb3=='py') and S_Cu_12==1 :
                     wgt_d10Ld8[1]+=abs(vecs[istate,k])**2      
                if orb1=='dx2y2' and  orb2=='d3z2r2'  and  (orb3=='px' or orb3=='py') and S_Cu_12==0 :
                     wgt_d10Ld8[2]+=abs(vecs[istate,k])**2                     
                    
            elif (orb1=='px' or orb1=='py') and  (orb2=='px' or orb2=='py')  and  (orb3=='px' or orb3=='py') :
                 wgt_d10d10[0]+=abs(vecs[istate,k])**2             
    
               
                
            sumweight=wgt_d9d8[0]+wgt_d8d9[0]+wgt_d9d9L[0]+wgt_d9d10L2[0]+wgt_d10d9L2[0]+wgt_d9L2d10[0]+wgt_d10Ld9L[0]+wgt_d9Ld10L[0]\
                      +wgt_d10L2d9[0]+wgt_d10d8L[0]+wgt_d9Ld9[0]+wgt_d8d10L[0]+wgt_d8Ld10[0]+wgt_d10Ld8[0]+wgt_d10d10[0]

    print ('sumweight=',sumweight/number)
    print ('sumweight1=',sumweight1/number)
    print ('wgt_d9d8=',wgt_d9d8[0]/number)
    print ('wgt_d8d9=',wgt_d8d9[0]/number)
    print ('wgt_d9d9L=',wgt_d9d9L[0]/number) 
    print ('wgt_d9d10L2=',wgt_d9d10L2[0]/number)
    print ('wgt_d10d9L2=',wgt_d10d9L2[0]/number)
    print ('wgt_d9L2d10=',wgt_d9L2d10[0]/number) 
    print ('wgt_d10Ld9L=',wgt_d10Ld9L[0]/number)
    print ('wgt_d9Ld10L=',wgt_d9Ld10L[0]/number)
    print ('wgt_d10L2d9=',wgt_d10L2d9[0]/number)       
    print ('wgt_d10d8L',wgt_d10d8L[0]/number)
    print ('wgt_d9Ld9=',wgt_d9Ld9[0]/number)
    print ('wgt_d8d10L=',wgt_d8d10L[0]/number)
    print ('wgt_d8Ld10=',wgt_d8Ld10[0]/number)
    print ('wgt_d10Ld8=', wgt_d10Ld8[0]/number) 
    print ('wgt_d10d10 =',wgt_d10d10[0]/number)        
    
        
#         print ('H=',wgt_H[0]) 

     
#         print ('s11=',s11)        
#         print ('s10=',s10)       
#         print ('s01=',s01)  
#         print ('s00=',s00)          


    path = './data'		# create file

    if os.path.isdir(path) == False:
        os.mkdir(path) 
        
    txt=open('./data/d9d8','a')                                  
    txt.write(str(wgt_d9d8[0]/number)+'\n')
    txt.close()            
    txt=open('./data/d9d8_b1b1b1','a')                                  
    txt.write(str(wgt_d9d8[1]/number)+'\n')
    txt.close()            
    txt=open('./data/d9d8_b1a1b1_1','a')                                  
    txt.write(str(wgt_d9d8[2]/number)+'\n')
    txt.close()              
    txt=open('./data/d9d8_a1b1b1','a')                                  
    txt.write(str(wgt_d9d8[3]/number)+'\n')
    txt.close()  
    
    txt=open('./data/d8d9','a')                                  
    txt.write(str(wgt_d8d9[0]/number)+'\n')
    txt.close()           
    txt=open('./data/d8d9_b1b1b1','a')                                  
    txt.write(str(wgt_d8d9[1]/number)+'\n')
    txt.close()          
    txt=open('./data/d8d9_b1b1a1','a')                                  
    txt.write(str(wgt_d8d9[2]/number)+'\n')
    txt.close()     
    txt=open('./data/d8d9_a1b1b1_1','a')                                  
    txt.write(str(wgt_d8d9[3]/number)+'\n')
    txt.close()        
        
    txt=open('./data/d9d9L','a')                                  
    txt.write(str(wgt_d9d9L[0]/number)+'\n')
    txt.close()          
    txt=open('./data/d9d9L_b1b1','a')                                  
    txt.write(str(wgt_d9d9L[1]/number)+'\n')
    txt.close()    
    txt=open('./data/d9d9L_b1a1','a')                                  
    txt.write(str(wgt_d9d9L[2]/number)+'\n')
    txt.close()       
    txt=open('./data/d9d9L_a1a1','a')                                  
    txt.write(str(wgt_d9d9L[3]/number)+'\n')
    txt.close() 
    
    txt=open('./data/d9d10L2','a')                                  
    txt.write(str(wgt_d9d10L2[0]/number)+'\n')
    txt.close()    
    txt=open('./data/d9d10L2_b1','a')                                  
    txt.write(str(wgt_d9d10L2[1]/number)+'\n')
    txt.close()        
    txt=open('./data/d9d10L2_a1','a')                                  
    txt.write(str(wgt_d9d10L2[2]/number)+'\n')
    txt.close()            
    
    txt=open('./data/d10d9L2','a')                                  
    txt.write(str(wgt_d10d9L2[0]/number)+'\n')
    txt.close()     
    txt=open('./data/d10d9L2_b1','a')                                  
    txt.write(str(wgt_d10d9L2[1]/number)+'\n')
    txt.close()         
    txt=open('./data/d10d9L2_a1','a')                                  
    txt.write(str(wgt_d10d9L2[2]/number)+'\n')
    txt.close()     
    
    txt=open('./data/d10Ld9L','a')                                  
    txt.write(str(wgt_d10Ld9L[0]/number)+'\n')
    txt.close()    
    txt=open('./data/d10Ld9L_b1','a')                                  
    txt.write(str(wgt_d10Ld9L[2]/number)+'\n')
    txt.close()        
    txt=open('./data/d10Ld9L_a1','a')                                  
    txt.write(str(wgt_d10Ld9L[3]/number)+'\n')
    txt.close()        
    
    txt=open('./data/d10d8L','a')                                  
    txt.write(str(wgt_d10d8L[0]/number)+'\n')
    txt.close() 
    txt=open('./data/d10d8L_a1b1_s1','a')                                  
    txt.write(str(wgt_d10d8L[1]/number)+'\n')
    txt.close()     
    txt=open('./data/d10d8L_a1b1_s0','a')                                  
    txt.write(str(wgt_d10d8L[2]/number)+'\n')
    txt.close()         
    
    txt=open('./data/d8d10L','a')                                  
    txt.write(str(wgt_d8d10L[0]/number)+'\n')
    txt.close()     
    txt=open('./data/d8d10L_a1b1_s1','a')                                  
    txt.write(str(wgt_d8d10L[1]/number)+'\n')
    txt.close()         
    txt=open('./data/d8d10L_a1b1_s0','a')                                  
    txt.write(str(wgt_d8d10L[2]/number)+'\n')
    txt.close()         
    
    txt=open('./data/d10d10','a')                                  
    txt.write(str(wgt_d10d10[0]/number)+'\n')
    txt.close()      
    
    
 
    print("--- get_ground_state %s seconds ---" % (time.time() - t1))
                
    return vals, vecs 
