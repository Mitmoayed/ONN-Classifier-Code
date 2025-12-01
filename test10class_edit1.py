import numpy as np
import cv2
import os
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import concurrent.futures
from joblib import Parallel, delayed
import time
import pandas as pd
import numpy as np
import cv2
import os
from numba import jit, float64
from numpy import inf
from scipy.spatial.distance import hamming
import numpy as np
from skimage import img_as_float
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga
from scipy import stats
import sklearn.metrics as sklm
from collections import Counter
#new add Begin######################################################################################################################
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#N = 49
N = 36
#new add end########################################################################################################################
number_of_class=2
t = np.arange(0, 10.1, 0.1)
testData=None
retOHNN2=None

np.set_printoptions(precision=15)
pd.set_option('display.precision', 15)

def draw81without (x):
    plt.figure(figsize=(4,4))
    #new add Begin######################################################################################################################
    #1Line
    #new add end########################################################################################################################
    plt.imshow(x.reshape((6,6)), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #plt.imshow(x.reshape((12,12)), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #if title:
    #plt.title(title)
    plt.axis('off')
    #if save_path:
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw81 (x):
    plt.figure(figsize=(4,4))
    #new add Begin######################################################################################################################
    #1Line
    #new add end########################################################################################################################
    plt.imshow(x.reshape((6,6), order='F'), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #plt.imshow(x.reshape((12,12), order='F'), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    #if title:
    #plt.title(title)
    plt.axis('off')
    #if save_path:
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw4(th,phi2,t_out,i):
    plt.figure(figsize=(8, 5))
    plt.plot(t_out, np.sin(th), linewidth=1.5)
    plt.title('sin(theta) vs Time'+str(i))
    plt.xlabel('Time')
    plt.ylabel('sin(theta)')
    plt.grid(True)
    plt.show()

    # --- بخش دوم: نمایش اختلاف فاز ---
    oscillator_1 = phi2[:, 0]  # فاز نوسانگر اول (رفرنس)
    phase_differences = (oscillator_1[:, None] - phi2) % (2 * np.pi)
    phase_differences = np.unwrap(phase_differences, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(t_out, phase_differences, linewidth=1.5)
    plt.title('Phase Difference of Oscillator 1 with All Oscillators')
    plt.xlabel('Time')
    plt.ylabel('Unwrapped Phase Difference (radians)')
    plt.grid(True)
    plt.show()

def imbinarize_global(I, level=None):
    # convert image to float (like MATLAB)
    I = img_as_float(I)
    
    # if level is not provided, use mean as a simple threshold (MATLAB default behavior can vary)
    if level is None:
        level = np.mean(I)
    else:
        level = float(level)
    
    # binarize the image
    BW = (I >= level).astype(np.uint8)
    
    return BW
def hebb_correct(x):
    P, L = x.shape
    W = np.zeros((L, L))
    
    for p in range(P):
        # تبدیل بردار 1D به ستون 2D برای ضرب درست
        #v = x[p, :].reshape(L, 1)
        v = x[p, :].flatten(order='F')[:,np.newaxis]
        

        W += np.dot(v, v.T)
    
    np.fill_diagonal(W, 0)
    W = W / 2  # تقسیم مثل کد متلب
    W = W - (P / (L**10))
    return W



    
def ONN_ode(theta, t, W, frq):   #for odeient solver
    #print(theta, t)
   
    theta = np.atleast_1d(theta)  # مطمئن می‌شویم که 1D است
    N = len(theta)
    thetaDOT = np.zeros(N)
    
    for i in range(N):
        thetaDOT[i] = frq + np.sum(W[i, :] * np.sin(theta - theta[i]))
    #print(thetaDOT)
    #pd.DataFrame(thetaDOT).to_csv(r'C:\Users\Mitra\Desktop\New\thetaDot.csv')

    return thetaDOT

@jit(nopython=True)
def ONN(t, theta, W, frq):  #for Rk45 solver
    #print(theta)
    #theta = np.atleast_1d(theta)  # مطمئن شو 1D
    #print(theta)
    
    N = len(theta)
    thetaDOT = np.zeros(N)

    for i in range(N):
        thetaDOT[i] = frq + np.sum(W[i, :] * np.sin(theta - theta[i]))

    return thetaDOT   # باید (N,) باشه
    #return thetaDOT.reshape(-1, 1)

counteri=0
def process_test_category(args):
    test_idx, category_idx, t, testData, W_all, frq = args
    W_category = W_all[:, :, category_idx]
    #print(W_all)
    #print(W_category)
    #pd.DataFrame(W_all[:, :, category_idx]).to_csv(r'C:\Users\Mitra\Desktop\New\W_all___new.csv')
    #theta = odeint(ONN, testData[test_idx, :], t, args=(W_category, frq), rtol=1e-10, atol=1e-12)
    y0 = testData[test_idx, :]
    #print(testData)
    #pd.DataFrame(testData).to_csv(r'C:\Users\Mitra\Desktop\New\testdata.csv')
    #print(testData.shape)
    
    y0 = np.ravel(y0)  # مطمئن می‌شویم 1D است
    #theta = odeint(ONN_ode, y0.astype(np.float64), t, args=(W_category.astype(np.float64), frq), rtol=1e-10, atol=1e-12)
    #theta = odeint(ONN, y0.astype(np.float64), t, args=(W_category.astype(np.float64), frq))
    
    #draw81(y0)

    def ONN_wrapper(t, theta, W_category, frq):
        return ONN(t, theta, W_category, frq)

    sol = solve_ivp(
        fun=ONN,
        t_span=(t[0], t[-1]),
        y0=y0.astype(np.float64),
        #method='DOP853',
        #method='BDF',
        method='RK45',
        t_eval=t,
        args=(W_category.astype(np.float64), frq),
        rtol=1e-8,
        atol=1e-6,
        #dense_output=True,
        dense_output=False,
    )
    theta = sol.y.T
    #print(t)
    #print(frq)
    #oooooo
    
    #theta = rk4_ONN(ONN, t, y0.astype(np.float64),
            #W_category.astype(np.float64), frq)

 

    #theta=np.round(theta,4)

    #pd.DataFrame(theta).to_csv(r'C:\Users\Mitra\Desktop\New\theta.csv')
    global counteri
    #theta=pd.read_csv(r"C:\Users\Mitra\Desktop\New\theta_matlab"+str(counteri+1)+".csv").to_numpy()
    #phi2=pd.read_csv(r"C:\Users\Mitra\Desktop\New\phi2_matlab"+str(counteri+1)+".csv").to_numpy()


    #pd.DataFrame(theta).to_csv(r'C:\Users\Mitra\Desktop\New\theta'+str(counteri)+'.csv')
    counteri+=1
    all_theta = np.zeros((len(testData), len(t), N))
    all_phi   = np.zeros((len(testData), len(t), N))
    
    phi2 = np.pi / 2 * (1 + np.sin(theta - frq * t.flatten(order='F')[:,np.newaxis]))#reshape((-1, 1))))
    all_theta[test_idx, :, :] = theta
    all_phi[test_idx, :, :] = phi2

    
    counteri += 1


    #phi2=np.round(phi2,4)
    #pd.DataFrame(phi2).to_csv(r'C:\Users\Mitra\Desktop\New\phi2.csv')
    #print(type(phi2))

    #print(phi2.dtype)
    #global counteri
    if counteri<20 and False:
     draw(theta,phi2,t,counteri)
     counteri+=1
    #print(phi2)
    #return (test_idx, category_idx, (phi2[-1, :] > np.pi/2) & (phi2[-1, :] < 1.5*np.pi))
    retOHNN2[test_idx, :, category_idx] = (phi2[-1, :] > np.pi/2) & (phi2[-1, :] < 1.5*np.pi)

    #tol = 0.2 # یک عدد بسیار کوچک به عنوان حاشیه اطمینان
    #retOHNN2[test_idx, :, category_idx] = (phi2[-1, :] > (np.pi/2 - tol)) & (phi2[-1, :] < (1.5*np.pi + tol))

    

#new add Begin######################################################################################################################
#1Line
#new add end########################################################################################################################
def fitness(combined_patterns,    first_class, secend_class,group1,group2,no_score=True):
    global retOHNN2
    combined_patterns = combined_patterns.reshape((number_of_class,N,1))
    # محاسبه ماتریس‌های وزن
    W_all = []
    #combined_patterns = combined_patterns[:, :, np.newaxis]   # حالا ابعادش میشه (2, 81, 1)

    for j in range(combined_patterns.shape[2]):

        # هر دسته از combined_patterns باید (m x n) باشد که m تعداد الگوها و n تعداد نورون‌ها است
        W_current = hebb_correct((combined_patterns[:, :, j]*2)-1)
        W_all.append(W_current)
        #print(f"دسته {j+1}: ابعاد ماتریس وزن = {W_current.shape}")
    #print(W_all)
    W_all = np.dstack(W_all)
    #print(W_all.shape)
    print(f"W_all: {W_all.shape}")

    W_to_save = W_all[:, :, 0]
    #pd.DataFrame(W_to_save).to_csv(r'C:\Users\Mitra\Desktop\New\W.ALL.csv')
    #W_all=pd.read_csv(r'C:\Users\Mitra\Desktop\New\W_all_matlab.csv',header=None,dtype=np.float64).to_numpy().reshape((81,81,1))
    #print(W_all)


    # Test Data Preparation
    
    #new add Begin######################################################################################################################
    T =400
    testData0   = None
    true_labels = None
    if False and train:
        for i in group1+group2:
            mask = (y_train == i) 
            if i in group1:
                X = x_train[mask][:T//10]
                y = y_train[mask][:T//10]
            else:
                X = x_train[mask][:T//10]
                y = y_train[mask][:T//10]
            if testData0 is None:
                testData0   = X.copy()
                true_labels = y.copy()
            else:
                testData0   = np.concatenate((testData0,X),axis=0)
                true_labels = np.concatenate((true_labels,y),axis=0)
    else:         
        '''for i in group1+group2 if no_score else [first_class,secend_class] :#
            #print(i)
            
            mask = (y_test == i)
            if no_score:    
                X = x_test[mask]#[:892]
                y = y_test[mask]#[:892]
            else:    
                X = x_test[mask][:892]
                y = y_test[mask][:892]
    
            if testData0 is None:
                testData0   = X.copy()
                true_labels = y.copy()
            else:
                testData0   = np.concatenate((testData0,X),axis=0)
                true_labels = np.concatenate((true_labels,y),axis=0)'''
        testData0=x_test.copy()
        true_labels=y_test.copy()

    if len(group1)==5 and len(group2)==5:
     
     true_labels[np.in1d(true_labels, group2)] = secend_class
     #true_labels[np.in1d(true_labels,group1)] = first_class   
     true_labels[np.in1d(true_labels, group1)] = first_class 
     #true_labels[np.in1d(true_labels,group2)] = secend_class  
     
    #print(Counter(np.ravel(true_labels)))  
    
     #print('mitra') 
    T = testData0.shape[0]   
    testData = np.zeros((T, N))
    for i in range(testData.shape[0]):
        testData[i]    = cv2.resize(testData0[i][2:-2,2:-2], (6,6)).reshape((N))#, interpolation=cv2.INTER_LINEAR
        testData[i]    = imbinarize_global(testData[i]) 
        testData[i]    = testData[i]*2-1 
        testData[i]    = testData[i]*(np.pi/2) 

    #draw81without (testData[5])  
    #draw81without (testData[-8])    
    #new add end########################################################################################################################





    # Optimization Loop
    best_accuracy = 0
    best_params = {'frq': 0, 't_end': 0, 'category': 0}
    frq_values = [1.0]
    t_values = [50]
    best_confusion_labels = None
    ################################################################################################
    # Create a helper function for parallel processing
    '''def rk4_ONN(ONN, t, y0, W_category, frq):
        """
        ONN        : تابع دینامیک شبکه (dy/dt)
        t          : آرایه زمان‌ها (np.linspace)
        y0         : بردار اولیه (1D numpy)
        W_category : ماتریس وزن (N, N)
        frq        : فرکانس (اسکالر)
        """
        h = t[1] - t[0]                     # اندازه گام
        y = np.zeros((len(t), len(y0)))     # ذخیره نتایج
        y[0] = y0                           # مقدار اولیه
        
        for i in range(1, len(t)):
            ti = t[i-1]        # زمان فعلی
            yi = y[i-1]        # مقدار حالت فعلی
            
            # محاسبه مشتقات (k1...k4)
            k1 = ONN(ti, yi, W_category, frq)
            k2 = ONN(ti + h/2, yi + h/2*k1, W_category, frq)
            k3 = ONN(ti + h/2, yi + h/2*k2, W_category, frq)
            k4 = ONN(ti + h, yi + h*k3, W_category, frq)
            
            # آپدیت مقدار بعدی
            y[i] = yi + h/6*(k1 + 2*k2 + 2*k3 + k4)
        
        return y'''





        
    #mnnn
    '''print("testData.shape:", testData.shape)
    print("testData[0] =", testData[0])
    print("W_all.shape:", W_all.shape)
    print("W_all[:,:,0] =", W_all[:,:,0])
    print("test_idx =", test_idx)
    kkkkkkkk'''
    '''for i in range (10):
     draw81(testData[i,:])'''
    predicted_label_list=[]
    for frq in frq_values:
        for t_end in t_values:
            #t = np.arange(0, 50.1 , 0.1)
            #t = np.linspace(0, 0.1, 500)
            #t = np.linspace(0, t_end, 1000)   # مثلا 1000 نقطه بین 0 و t_end
            #t = np.linspace(0, 50, 5000)  # به جای 0 تا 10 با 100 نقطه
            #new add Begin######################################################################################################################
            #1Line
            #new add end########################################################################################################################
            retOHNN2 = np.zeros((T, N, combined_patterns.shape[2]))
            
            # Parallel processing
            tasks = []
            results=[]
            for category_idx in range(combined_patterns.shape[2]):
                for test_idx in range(T):
                    tasks.append((test_idx, category_idx, t, testData, W_all, frq))
                    process_test_category(tasks[-1])
            #results = Parallel(n_jobs=-1)(delayed(process_test_category)(task) for task in tasks)
            #pd.DataFrame(results).to_csv(r'C:\Users\Mitra\Desktop\New\results.csv')
            #results = process_test_category(tasks[0])
            #result5=pd.read_csv(r"C:\Users\Mitra\Desktop\New\retOHNN2.csv").to_numpy()
            #for test_idx, category_idx, result in results:
            #retOHNN2[test_idx, :, category_idx] = result
            #retOHNN2=pd.read_csv(r"C:\Users\Mitra\Desktop\New\retonnmat.csv",header=None).to_numpy()[ :,:,np.newaxis]
            #print(retOHNN2.shape)
            #pd.DataFrame(retOHNN2.reshape((T,81))).to_csv(r'C:\Users\Mitra\Desktop\New\retonnpy.csv')
            #print(retOHNN2.shape)

            #pd.DataFrame(retOHNN2.reshape((T,81))).to_csv(r'C:\Users\Mitra\Desktop\New\retOHNN2.csv')
            # Accuracy calculation
            accuracy_per_category = np.zeros(combined_patterns.shape[2])
            predicted_labels_all = np.zeros((T, combined_patterns.shape[2]))
            
            for category_idx in range(combined_patterns.shape[2]):
                current_patterns = combined_patterns[:, :, category_idx]
                #print(current_patterns)
                #pd.DataFrame(current_patterns.reshape((2,81))).to_csv(r'C:\Users\Mitra\Desktop\New\compattpy.csv')
                current_retOHNN2 = retOHNN2[:, :, category_idx]
                #pd.DataFrame(retOHNN2.reshape((T,81))).to_csv(r'C:\Users\Mitra\Desktop\New\retOHNN2py.csv')
                
                correct = 0
                #print(current_retOHNN2.shape)
               
                for test_idx in range(T):
                    min_hamming_dist =inf #float('inf')
                    #print(min_hamming_dist)
                
                    best_match = -1
                    #draw81(current_retOHNN2[test_idx,:])
                    #draw81(current_patterns[0,:])
                    #draw81(current_patterns[1,:])
                    for i in range(current_patterns.shape[0]):
                        hamming_dist = np.sum(current_retOHNN2[test_idx, :] != current_patterns[i, :])
                        hamming_dist_inv = np.sum(current_retOHNN2[test_idx, :] != 1- current_patterns[i, :])
                        min_dist = min(hamming_dist, hamming_dist_inv)
                        
                        #print(1 - current_patterns[i, :])
                        #print(current_patterns[i, :])
                        
                        if min_dist < min_hamming_dist:
                            min_hamming_dist = min_dist
                            best_match = i
                    
                    if best_match == -1:
                        predicted_label = np.nan
                    else:
                        if best_match in [0]:  # Adjust based on your pattern indices
                            predicted_label = first_class
                        else:
                            #predicted_label = 1
                            predicted_label = secend_class
                    predicted_label_list.append(predicted_label)
                    predicted_labels_all[test_idx, category_idx] = predicted_label
                    #print(predicted_label_list)
                    #print(predicted_labels_all)
                    if predicted_label == true_labels[test_idx]:
                        #print(len(true_labels))
                        correct += 1
                
                accuracy_per_category[category_idx] = (correct / T) * 100
                #print(f'Frequency: {frq} | Time: {t_end} | Training Set {category_idx+1}: Accuracy = {accuracy_per_category[category_idx]:.2f}%')
            #print(predicted_labels_all)
            # Update best parameters
            current_max_accuracy = np.max(accuracy_per_category)
            current_best_category = np.argmax(accuracy_per_category)
            
            if current_max_accuracy > best_accuracy:
                best_accuracy = current_max_accuracy
                best_params['frq'] = frq
                best_params['t_end'] = t_end
                best_params['category'] = current_best_category
                best_confusion_labels = predicted_labels_all[:, current_best_category]

    # Display results
    #print(f'\n=== Best Parameters ===')
    #print(f'Frequency: {best_params["frq"]}')
    #print(f'Time: {best_params["t_end"]}')
    #print(f'Best Category: {best_params["category"]+1}')
    #print(f'Accuracy: {best_accuracy:.2f}%')

    # Confusion matrix
    if best_confusion_labels is not None and False:
        print(len(true_labels),len(predicted_label_list))
        cm = confusion_matrix(true_labels, predicted_label_list)#best_confusion_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 2'])
        disp.plot()
        plt.title(f'Confusion Matrix (Best Accuracy: {best_accuracy:.2f}%)')
        plt.show()

    # Save results
    #np.save('ONN2_zerohori_final.npy', retOHNN2)
    #np.save('current_patterns2.npy', combined_patterns)
    #print('Results saved successfully.')


    '''def draw(theta, phi2, t, i, testData, img_shape=(9, 9)):
        # نمایش تصویر تست
        plt.figure(figsize=(3, 3))
        img = testData[i, :].reshape(img_shape, order='F')# reshape بر اساس اندازه‌ی تصویر
        plt.imshow(img, cmap='gray')
        plt.title(f"Test Image {i+1}")
        plt.axis("off")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(t, np.sin(theta), linewidth=1.5)
        plt.title(f'sin(\\theta) vs Time for Sample {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('sin(\\theta)')
        plt.grid(True)
        plt.show()

        oscillator_1 = phi2[:, 0]
        
        #phase_differences = np.unwrap(np.mod(oscillator_1 - phi2, 2 * np.pi), axis=0)
        phase_differences = np.unwrap(np.mod(oscillator_1[:, np.newaxis] - phi2, 2 * np.pi), axis=0)
        plt.figure(figsize=(10, 5))
        plt.plot(t, phase_differences, linewidth=1.5)
        plt.title(f'Phase Difference of Oscillator 1 with All Oscillators - Sample {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Unwrapped Phase Difference (radians)')
        plt.grid(True)
        plt.show()

    # حلقه برای نمایش
    for i in range(len(testData)):
        theta = all_theta[i, :, :]  # استخراج theta
        phi2 = all_phi[i, :, :]     # استخراج phi2
        draw(theta, phi2, t, i, testData, img_shape=(9, 9))'''
    print("-----",len(predicted_label_list),T)   
    
    out = np.array(predicted_label_list).reshape((-1,1))
    
    
    acc1 = sklm.accuracy_score(true_labels, np.ravel(out))
    new_out = out.copy()
    new_out[new_out==first_class]=11
    new_out[new_out==secend_class]=first_class
    new_out[new_out==11]=secend_class
    acc2 = sklm.accuracy_score(true_labels, np.ravel(new_out))
    
    if acc1>acc2:
        print("new_best_accuracy",acc1)
        return out
    else:
        print("new_best_accuracy",acc2)
        return new_out


def most_frequent_per_row_mode(matrix):
    """پیدا کردن پرتکرارترین عنصر هر ردیف با استفاده از mode"""
    if matrix.size == 0:
        return None
    
    # پیدا کردن mode برای هر ردیف
    mode_result = stats.mode(matrix, axis=1)
    
    # mode_result[0] = مقادیر پرتکرار، mode_result[1] = تعداد تکرار
    return mode_result[0].flatten(), mode_result[1].flatten()


group=[[0,1,2,3,4,5,6,7,8,9]]
#group=[[0,1]]
#best_sloution = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

#index_partiton = fitness(np.array(best_sloution),    0,1,group[0],group[1])


score = [] 
for l in range(1):
    y_test_fix = y_test#[(y_test==0)|(y_test==1)]#[np.ravel(index_partiton)==l]
    #print(Counter(np.ravel(index_partiton)==l))
    #print(Counter(np.ravel(index_partiton)))
    
    best_sloutions=pd.read_csv(r'c:\users\Mitra\Desktop\New\bestsloution'+str(l+1)+'ndpart1.csv')
    #best_sloutions = pd.read_csv('/home/mitra/Python/bestsloution' + str(l+1) + 'ndpart1.csv') #linux file
    
    rslt_list = None
    if True:
     for i in range (len(group[l])):
        for j in range (i+1,len(group[l])):
            if i!= j:
                print("="*20,group[l][i],group[l][j])
                best_sloution = eval(best_sloutions.iloc[j,i+1].replace('\n','').replace('. ',',').replace('.',''))
                rslt = fitness(np.array(best_sloution),    group[l][i],group[l][j],group[l],[])
                if rslt_list is None:
                    rslt_list = rslt.copy()
                else:
                    rslt_list = np.concatenate((rslt_list, rslt),axis=-1)
     #print(rslt_list.shape)
     #print(rslt_list.shape, len(y_test_fix))

    pd.DataFrame(rslt_list).to_csv(r"C:\Users\Mitra\Desktop\New\matrix_prediction"+str(l+1)+".csv")                
    most_frequent_elements, counts = most_frequent_per_row_mode(rslt_list)
    '''for i in range(10):
    
        print(y_test[y_test==i].shape)
        if y_test_fix is None:
            y_test_fix = y_test[y_test==i]#[:892]
        else:    
            y_test_fix = np.concatenate((y_test_fix,y_test[y_test==i]))'''
    #most_frequent_elements=pd.read_csv(r"C:\Users\Mitra\Desktop\New\modes_output.csv")
    score.append(sklm.accuracy_score(most_frequent_elements,y_test_fix))
    #print(Counter(counts))
    print("score",l,score[-1])
    #print(Counter(counts))
print("finall score",np.mean(score[-1]))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as sklm  # اگر قبلاً import نکردی

# فرض کنیم cm قبلاً محاسبه شده (از کدت)
cm = sklm.confusion_matrix(y_test_fix,most_frequent_elements)

# پلات heatmap (رنگ‌ها بر اساس مقادیر، اعداد رو هم نشون می‌ده)
plt.figure(figsize=(8, 6))  # اندازه فیگور رو تنظیم کن
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)  # annot=True برای نشون دادن اعداد، cmap='Blues' برای رنگ آبی
class_names = ['0','1','2','3','4','5','6','7','8','9']  # نام کلاس‌هات
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')  # عنوان
plt.xlabel('Predicted Labels')  # برچسب x
plt.ylabel('True Labels')  # برچسب y
plt.show()  # نمایش پلات
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# فرض: y_true و y_pred خروجی‌های واقعی مدل هستند
# آن‌ها را از دیتاست یا مدل خودت بگیر
# مثال:
# y_true = [0, 1, 2, 1, 0, 3, 4, 5, 6, 7, 8, 9, ...]
# y_pred = [0, 1, 2, 0, 0, 3, 4, 5, 6, 7, 8, 8, ...]

# محاسبه ماتریس گیجی
#cm = confusion_matrix(y_true, y_pred)

# استخراج تعداد درست و خطا برای هر کلاس


correct_counts = cm.diagonal()
total_counts = cm.sum(axis=1)
errors = total_counts - correct_counts

# رسم هیستوگرام
classes = np.arange(len(correct_counts))  # کلاس‌ها بر اساس تعداد واقعی

plt.figure(figsize=(8,6))
plt.bar(classes, correct_counts, color="#1f77b4", alpha=0.8, label="Correct")
plt.bar(classes, errors, bottom=correct_counts, color="#d62728", alpha=0.8, label="Errors")

plt.xlabel("Class", fontsize=12)
plt.ylabel("Number of Samples", fontsize=12)
plt.title("Histogram of Correct vs Errors per Class", fontsize=14)
plt.xticks(classes)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

# ذخیره تصویر برای مقاله
plt.savefig("confusion_histogram.png", dpi=300, bbox_inches='tight')

plt.show()
accuracy_per_class = correct_counts / total_counts * 100

plt.figure(figsize=(8,6))
plt.plot(classes, accuracy_per_class, marker='o', color='green')
plt.xlabel("Class", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Per-Class Accuracy", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

correct_percent = correct_counts / total_counts * 100
error_percent = errors / total_counts * 100

plt.figure(figsize=(8,6))
plt.bar(classes, correct_percent, color="#1f77b4", alpha=0.8, label="Correct (%)")
plt.bar(classes, error_percent, bottom=correct_percent, color="#d62728", alpha=0.8, label="Errors (%)")
plt.xlabel("Class", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.title("Normalized Histogram of Correct vs Errors", fontsize=14)
plt.xticks(classes)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))
plt.pie([sum(correct_counts), sum(errors)],
        labels=["Correct", "Errors"],
        colors=["#1f77b4", "#d62728"],
        autopct='%1.1f%%',
        startangle=90)
plt.title("Overall Classification Accuracy")
plt.tight_layout()
plt.show()
