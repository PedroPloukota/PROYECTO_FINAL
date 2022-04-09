import numpy as npy
import matplotlib.pyplot as plt
import seaborn as sns



size_gen = 300
m=1
b=2


sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

def generar_listado_a(sizes):
    return list(npy.random.randint(low = 1,high=sizes,size=sizes))

def generar_listado_b(sizes):
    #return list(npy.random.poisson(800, size_gen))
    return npy.random.normal(sizes, sizes*0.8, size=(sizes))

def generar_listado_c(sizes):
    return list(npy.random.poisson(sizes, sizes))


def generar_listado_y_hat(data_z):
    data_ex=[]
    for n in data_z:     
        data_ex.append(n*m+b)              
    return data_ex
     
def generar_listado_error(data_t,data_r):
    error=[]
    for n in range(data_t.size):  
        error.append(((data_r[n]-data_t[n])**2)/2)            
    return error


def generar_listado_gradiente_m(data_x,data_w,data_q):
    gradiente_xc=[]
    for n in range(data_x.size):         
        gradiente_xc.append((data_q[n]-data_w[n])*data_x[n])            
    return gradiente_xc

def generar_listado_gradiente_b(data_f,data_g):
    gradiente_xz=[]
    for n in range(data_f.size):         
        gradiente_xz.append((data_g[n]-data_f[n]))            
    return gradiente_xz     

def graficar_univariable(function):
    import matplotlib.pyplot as plt_t

    plt_t.plot(function)    
    plt_t.show() 

def graficar_univariable_doble(function_B,function_A):
    import matplotlib.pyplot as plt_t
    plt_t.plot(function_B) 
    plt_t.plot(function_A)    
    plt_t.show() 
    
    
def skylear_mask():
     from sklearn import svm, datasets
     from sklearn.model_selection import cross_val_score

     return 0   

def genera_listado():
    #cada rango tiene 6 celdas se multiplica por 6 la cantidad
    cantidad = 7008
    function_A=[]  
    data_b=[]
    data_b=npy.load('proyecto_training_data.npy')
    
    data_b=data_b[:cantidad].flatten();
    data_b = npy.array([x for x in data_b if npy.isnan(x) == False])

    
    npy.save('A', npy.array(generar_listado_a(data_b.size)))
    #npy.save('B', npy.array(generar_listado_b()))
   
 
    data_a=npy.load('A.npy')
    

    #print(data_b)

    sns.scatterplot(data_a, data_b)

    data_y=generar_listado_y_hat(data_a)

    
    error_promedio = npy.mean(generar_listado_error(data_b,data_y))
    print('Error_promedio_A:',error_promedio)  
    
    gradiente_m = generar_listado_gradiente_m(data_a,data_b,data_y)
    gradiente_b = generar_listado_gradiente_b(data_b,data_y)

    
    gb_prom=npy.mean(gradiente_b)
    gm_prom=npy.mean(gradiente_m)


    for x in range(10): 
        x*gm_prom+gb_prom
        function_A.append(x*gm_prom+gb_prom)

    return(function_A)

def operar_listado_doble():
  cantidad = 7008
  A_function=[]  
  function_B=[] 

  data_b=npy.load('proyecto_training_data.npy')
  
  
  data_d=npy.load('proyecto_training_data.npy')

  data_b=data_b[:cantidad].flatten();
  data_d=data_d[:cantidad].flatten();
  
  data_b = npy.array([x for x in data_b if npy.isnan(x) == False])
  data_d = npy.array([x for x in data_d if npy.isnan(x) == False])
  
  print(data_b.size)
  
  npy.save('A', npy.array(generar_listado_a(data_d.size)))
  
  npy.save('C', npy.array(generar_listado_c(data_d.size)))
  
  data_a=npy.load('A.npy')
  data_c=npy.load('C.npy')

  sns.scatterplot(data_a,data_b)
  sns.scatterplot(data_c,data_d)
  
  
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
  from sklearn.metrics import hamming_loss


    
  #plt.plot(function)  
  print('A) resultado de xm+b:')
  print(A_function)
  print('B) resultado de xm+b:')
  print(function_B)
  print('Media de error:')
  print(mean_absolute_percentage_error(A_function, function_B))



  
  return mean_absolute_percentage_error(A_function, function_B)


print('promedio de aprendisaje en error:',operar_listado_doble())

print('resultado de xm+b:',genera_listado())
#graficar_univariable(genera_listado())



