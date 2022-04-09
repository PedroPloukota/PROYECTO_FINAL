import numpy as npy
import matplotlib.pyplot as plt
import seaborn as sns



size_gen = 100
m=0
b=0


sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

def generar_listado_a():
    return list(npy.random.randint(low = 5,high=100,size=size_gen))

def generar_listado_b():
    #return list(npy.random.poisson(800, size_gen))
    return npy.random.normal(300, 100.5, size=(100))

def generar_listado_c():
    return list(npy.random.poisson(600, size_gen))


def generar_listado_y_hat(data_z):
    data_ex=[]
    for n in data_z:     
        data_ex.append(n*m+b)              
    return data_ex
     
def generar_listado_error(data_t,data_r):
    error=[]
    for n in range(size_gen):         
        error.append(((data_r[n]-data_t[n])**2)/2)            
    return error


def generar_listado_gradiente_m(data_x,data_w,data_q):
    gradiente_xc=[]
    for n in range(size_gen):         
        gradiente_xc.append((data_q[n]-data_w[n])*data_x[n])            
    return gradiente_xc

def generar_listado_gradiente_b(data_f,data_g):
    gradiente_xz=[]
    for n in range(size_gen):         
        gradiente_xz.append((data_g[n]-data_f[n]))            
    return gradiente_xz     


def genera_listado():
    function_A=[]  

    
    npy.save('A', npy.array(generar_listado_a()))
    npy.save('B', npy.array(generar_listado_b()))
 
    data_a=npy.load('A.npy')
    data_b=npy.load('B.npy')

    sns.scatterplot(data_a, data_b)

    
    
    data_y=generar_listado_y_hat(data_a)

     
    error_promedio = npy.mean(generar_listado_error(data_b,data_y))
    
    gradiente_m = generar_listado_gradiente_m(data_a,data_b,data_y)
    gradiente_b = generar_listado_gradiente_b(data_b,data_y)

    
    gb_prom=npy.mean(gradiente_b)
    gm_prom=npy.mean(gradiente_m)


    for x in range(10): 
        x*gm_prom+gb_prom
        function_A.append(x*gm_prom+gb_prom)
        
    import matplotlib.pyplot as plt_t

    print(function_A)
    plt_t.plot(function_A)    
    plt_t.show()

    return(function_A)

def operar_listado_doble():
  A_function=[]  
  function_B=[] 
  
  npy.save('A', npy.array(generar_listado_a()))
  npy.save('B', npy.array(generar_listado_b()))
  
  npy.save('C', npy.array(generar_listado_a()))
  npy.save('D', npy.array(generar_listado_c()))
  
  data_a=npy.load('A.npy')
  data_b=npy.load('B.npy')
  
  data_c=npy.load('C.npy')
  data_d=npy.load('D.npy')
   
  sns.scatterplot(data_a, data_b)
  sns.scatterplot(data_c, data_d)
  
  
  data_y=generar_listado_y_hat(data_a)
  data_y_c=generar_listado_y_hat(data_c)
  
  generar_listado_error(data_b,data_y) 
  generar_listado_error(data_d,data_y_c) 
 
    
  error_promedio_a = npy.mean(generar_listado_error(data_b,data_y)) 
  error_promedio_b = npy.mean(generar_listado_error(data_d,data_y_c)) 
    
  print('Error_promedio_A:',error_promedio_a)  
  print('Error_promedio_B:',error_promedio_b)  
  
  gradiente_m = generar_listado_gradiente_m(data_a,data_b,data_y)
  gradiente_b = generar_listado_gradiente_b(data_b,data_y)
  
  
  gradiente_mi = generar_listado_gradiente_m(data_c,data_d,data_y_c)
  gradiente_bi = generar_listado_gradiente_b(data_d,data_y_c)
  
  
  gb_prom=npy.mean(gradiente_b)
  gm_prom=npy.mean(gradiente_m)
  
  gb_prom_c=npy.mean(gradiente_bi)
  gm_prom_c=npy.mean(gradiente_mi)

  for x in range(10): 
      A_function.append(x*gm_prom+gb_prom)
  print('---------------------------')    
  for x in range(10): 
      function_B.append(x*gm_prom_c+gb_prom_c)      
      
  from sklearn.metrics import mean_absolute_percentage_error   
  
  mean_absolute_percentage_error(A_function, function_B)
    
  #plt.plot(function)   
  print(A_function)
  print(function_B)
  return mean_absolute_percentage_error(A_function, function_B)


print(operar_listado_doble())


print(genera_listado())
print(genera_listado())
print(genera_listado())




