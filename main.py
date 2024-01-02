import numpy as np
import matplotlib.pyplot as plt
import math

def transform(coords, parameters):
    Zoom_U, Zoom_V, Rot_U, Rot_V, Shift_U, Shift_V = parameters
    Rot_U = Rot_U * np.pi/180
    Rot_V = Rot_V * np.pi/180
    T = np.array(((Zoom_U * np.cos(Rot_U), Zoom_V * np.sin(Rot_U)), (Zoom_U * np.sin(Rot_V), Zoom_V * np.cos(Rot_V))))
    off = np.array(((Shift_U,),(Shift_V,)))

    coords_n = np.dot(T, coords) + off
    # coords_n = np.dot(T, coords) 
    return coords_n

def transform_high_order(coords, parameters_gen):
    # coords = np.vstack(x,y)
    # x = p_x[0] * x**0 * y**0 + p_x[1] * x**1 * y**0 + p_x[2] * x**0 * y**1 + p_x[3] * x**2 * y**0 + p_x[4] * x**1 * y**1 + p_x[5] * x**0 * y**2 + ...
    # y = p_y[0] * x**0 * y**0 + p_y[1] * x**1 * y**0 + p_y[2] * x**0 * y**1 + p_y[3] * x**2 * y**0 + p_y[4] * x**1 * y**1 + p_y[5] * x**0 * y**2 + ...
    p_x = parameters_gen[0,:]
    p_y = parameters_gen[1,:]

    coords_n = np.zeros(coords.shape)

    # x = p_x[0] * x**0 * y**0 + p_x[1] * x**1 * y**0 + p_x[2] * x**0 * y**1 + p_x[3] * x**2 * y**0 + p_x[4] * x**1 * y**1 +  + p_x[5] * x**0 * y**2 + ...

    for i in range(coords.shape[1]):
        for j in range(3 + 1):
            for k in range(j+1):
                # number of elements above a row inside pascal triangle
                # half of the square volume and half of the diagonal length
                # plus k
                index1 = int((j+1) * (j+1)/2 - (j+1)/2 + k)
                # pascal triangle contains this info itself on its 3rd diagonal
                # one row bellow
                # this is beacuse everything is constructed from ones
                # and adding them
                # if j>0:
                #     print(k + math.factorial(j + 1)/(math.factorial(2) * math.factorial(j-1)))
                #     # this is equvivalent to the above formula
                
                coords_n[0,i] += p_x[index1] * coords[0,i]**(j-k) * coords[1,i]**(k)
                coords_n[1,i] += p_y[index1] * coords[0,i]**(j-k) * coords[1,i]**(k)
                
    return coords_n

def get_para_3(coords_mask, coords_meas):
    coords_mask = coords_mask[:,:3]
    coords_meas = coords_meas[:3]
    A = np.vstack((coords_mask, np.ones(coords_mask.shape[1])))
    b = coords_meas
    return np.linalg.solve(A.T,b)

def get_para_LS(coords_mask, coords_meas):
    A = np.vstack((coords_mask, np.ones(coords_mask.shape[1])))
    ATA = np.dot(A,A.T)
    b = coords_meas
    ATb = np.dot(A,b)
    return np.linalg.solve(ATA,ATb)

def translate_params(y):
    # x_par[0]/x_par[1]
    # Zoom_U * np.cos(Rot_U)
    # Zoom_V * np.sin(Rot_U)
    # Zoom_U * np.sin(Rot_V)
    # Zoom_V * np.cos(Rot_V)
    # np.cos(Rot_U)/np.sin(Rot_V) = r_1/r_3
    # np.sin(Rot_U)/np.cos(Rot_V) = r_2/r_4

    # np.cos(Rot_U) = r_1/r_3 np.sin(Rot_V)
    # Rot_U = np.arccos(r_1/r_3 np.sin(Rot_V))

    # np.sin(np.arccos(r_1/r_3 np.sin(Rot_V)))/np.cos(Rot_V) = r_2/r_4
    # sqrt(1 - (r_1/r_3 np.sin(Rot_V))^2) = r_2/r_4 np.cos(Rot_V)
    # 1 - (r_1/r_3 np.sin(Rot_V))^2 = (r_2/r_4 np.cos(Rot_V))^2
    # 1 - (r_1/r_3)^2 * (1 - np.cos(Rot_V)^2) = (r_2/r_4 np.cos(Rot_V))^2
    r_1 = y[0]
    r_2 = y[1]
    r_3 = y[2]
    r_4 = y[3]

    # the derived formula has issues with large values
    # it can be complex, althouhg the imaginary part is everytime small and resulting just from the numerics
    if np.abs(r_1/r_3) > 1e10:
        Rot_V = 0
    else:
        Rot_V = (np.arccos((1 - (r_1/r_3)**2 + 0j)**(1/2)/((r_2/r_4)**2 - (r_1/r_3)**2 + 0j)**(1/2)))
        Rot_V = np.real(Rot_V)
    # also for negative angles it does give wrong sign, here correction
    if r_1/r_3<0:
        Rot_V = -1 * Rot_V

    Zoom_V = r_4/np.cos(Rot_V)
    Rot_U = np.arcsin(r_2/Zoom_V)
    Zoom_U = r_1/np.cos(Rot_U)
    return np.array((Zoom_U, Zoom_V, Rot_U/np.pi*180, Rot_V/np.pi*180, y[-2], y[-1]))

# def translate_params_NLS(x_par, y_par, x0):

    # Zoom_U * np.cos(Rot_U) = f1 = r1
    # Zoom_V * np.sin(Rot_U) = f2 = r2
    # Zoom_U * np.sin(Rot_V) = f3 = r3
    # Zoom_V * np.cos(Rot_V) = f4 = r4

    # df1/dZoom_U = np.cos(Rot_U)
    # df1/dZoom_V = 0
    # df1/dRot_U  = - Zoom_U * np.sin(Rot_U)
    # df1/dRot_V  = 0

    # df2/dZoom_U = 0
    # df2/dZoom_V = np.sin(Rot_U)
    # df2/dRot_U  = Zoom_V * np.cos(Rot_U)
    # df2/dRot_V  = 0

    # df3/dZoom_U = np.sin(Rot_V)
    # df3/dZoom_V = 0
    # df3/dRot_U  = 0
    # df3/dRot_V  = Zoom_U * np.cos(Rot_V)

    # df4/dZoom_U = 0
    # df4/dZoom_V = np.cos(Rot_V)
    # df4/dRot_U  = 0
    # df4/dRot_V  = - Zoom_V * np.sin(Rot_V)

    # Zoom_U = x0[0]
    # Zoom_V = x0[1]
    # Rot_U = x0[2]/180*np.pi
    # Rot_V = x0[3]/180*np.pi

    # x_1 = Zoom_U
    # x_2 = Zoom_V
    # x_3 = Rot_U
    # x_4 = Rot_V
    # x = np.array((x_1,x_2,x_3,x_4))
    # A = np.dot(Jf.T, Jf)
    # b = np.dot(Jf.T, r)
    # a = np.linalg.solve(A,b)

    # a[2] = a[2]/np.pi*180
    # a[3] = a[3]/np.pi*180

    # x[2] = x[2]/np.pi*180
    # x[3] = x[3]/np.pi*180

    # return x + a 


def Gaussian_Newton_for_NLS(Jacobian_fun, Residual_fun, beta_0, y, post_fun):

    Delta_tresh = 1e-10
    counter = 0
    Delta = 1
    beta = beta_0

    while (np.sum(Delta**2) > Delta_tresh) * (counter < 1e3):
        counter += 1
        Jf = Jacobian_fun(beta)
        r = Residual_fun(y, beta)
        A = np.dot(Jf.T, Jf)
        b = np.dot(Jf.T, r)
        Delta = np.linalg.solve(A,b)
        beta = beta + Delta

    residual = np.sum(r**2)
    beta = np.append(beta, np.array((y[-2],y[-1])))
    beta = post_fun(beta)
    return beta, residual, counter

def Unpack_beta_coord_translate(beta):
    Zoom_U = beta[0]
    Zoom_V = beta[1]
    Rot_U = beta[2]
    Rot_V = beta[3]

    return Zoom_U, Zoom_V, Rot_U, Rot_V

def Residual_for_coord_translate(y, beta):

    Zoom_U, Zoom_V, Rot_U, Rot_V = Unpack_beta_coord_translate(beta)

    r_1 = y[0] - Zoom_U * np.cos(Rot_U)
    r_2 = y[1] - Zoom_V * np.sin(Rot_U)
    r_3 = y[2] - Zoom_U * np.sin(Rot_V)
    r_4 = y[3] - Zoom_V * np.cos(Rot_V)
    r = np.array((r_1,r_2,r_3,r_4))

    return r

def Jacobian_for_coord_translate(beta):
    
    Zoom_U, Zoom_V, Rot_U, Rot_V = Unpack_beta_coord_translate(beta)
    
    df1a = np.cos(Rot_U)
    df1b = 0
    df1c = - Zoom_U * np.sin(Rot_U)
    df1d = 0

    df2a = 0
    df2b = np.sin(Rot_U)
    df2c = Zoom_V * np.cos(Rot_U)
    df2d = 0

    df3a = np.sin(Rot_V)
    df3b = 0
    df3c = 0
    df3d = Zoom_U * np.cos(Rot_V)

    df4a = 0
    df4b = np.cos(Rot_V)
    df4c = 0
    df4d = - Zoom_V * np.sin(Rot_V)
 
    Jf = np.array(((df1a , df1b, df1c, df1d),(df2a , df2b, df2c, df2d),(df3a , df3b, df3c, df3d),(df4a , df4b, df4c, df4d)))

    return Jf

def paramters_estimation_xy(coords_mask, coords_meas, estimator_fun):
    x_par = estimator_fun(coords_mask, coords_meas[0,:])
    y_par = estimator_fun(coords_mask, coords_meas[1,:])
    y = np.array((x_par[0], x_par[1], y_par[0], y_par[1], x_par[2], y_par[2]))
    return y

def paramters_rad_to_deg(beta):
    beta[2] *= 180/np.pi
    beta[3] *= 180/np.pi
    return beta


Shift_U = -10
Shift_V = 2
Zoom_U = 0.9
Zoom_V = 1.1
Rot_U = 10
Rot_V = 10

# Zoom_U = 1
# Zoom_V = 1
# Rot_U = 0
# Rot_V = 0


marker_scale = 1000
# 8 markers
U_mask = np.array((1,1,-1,-1, 1, -1, 0 , 0)) * marker_scale
V_mask = np.array((1,-1,1,-1, 0, 0 , 1, -1)) * marker_scale
# 4 markers
# U_mask = np.array((1,1,-1,-1)) * marker_scale
# V_mask = np.array((1,-1,1,-1)) * marker_scale
coords_mask = np.vstack((U_mask,V_mask))
sigma = 100
U_mask = U_mask + sigma * np.random.randn(U_mask.size)
V_mask = V_mask + sigma * np.random.randn(U_mask.size)
coords_mask_n = np.vstack((U_mask,V_mask))
# print(coords_mask_n)

parameters = [Zoom_U, Zoom_V, Rot_U, Rot_V, Shift_U, Shift_V]


coords_meas = transform(coords_mask_n, parameters)
y_3 = paramters_estimation_xy(coords_mask, coords_meas, get_para_3)
y_LS = paramters_estimation_xy(coords_mask, coords_meas, get_para_LS)

beta_0 = np.array((1,1,0,0,))

parameters_3_NLS, residual, interations = Gaussian_Newton_for_NLS(Jacobian_for_coord_translate, Residual_for_coord_translate, beta_0, y_3, paramters_rad_to_deg)
print("3_NLS_params, residual, iterations")
print((parameters_3_NLS), residual, interations)
parameters_LS_NLS, residual, interations = Gaussian_Newton_for_NLS(Jacobian_for_coord_translate, Residual_for_coord_translate, beta_0, y_LS, paramters_rad_to_deg)
print("3_NLS_params, residual, iterations")
print((parameters_LS_NLS), residual, interations)

parameters_3_direct = translate_params(y_3)
print("3_direct_params")
print(parameters_3_direct)
parameters_LS_direct = translate_params(y_LS)
print("LS_direct_params")
print(parameters_LS_direct)

# this is not so relevant, at first the parameters are non-linearly related to the coordinates
# also it depends on the poroportion of the writefield, respectively where the markers are
# better measure is from the coordinates
# print("3_NLS error from parameters")
# print(np.sum((parameters_3_NLS - parameters)**2))
# print("LS_NLS error from paramters")
# print(np.sum((parameters_LS_NLS - parameters)**2))
# print("their ratio")
# print(np.sum((parameters_3_NLS - parameters)**2)/np.sum((parameters_LS_NLS - parameters)**2))

U_mask = np.linspace(-1000,1000,11)
U, V = np.meshgrid(U_mask,U_mask)
coords_mask = np.vstack((U.flatten(),V.flatten()))


fig1, ax1 = plt.subplots()
coords_meas_r = transform(coords_mask, parameters)
ax1.plot(coords_meas_r[0,:],coords_meas_r[1,:], 'o')
coords_meas_r_m_x = np.reshape(coords_meas_r[0,:],(U_mask.size, U_mask.size))
coords_meas_r_m_y = np.reshape(coords_meas_r[1,:],(U_mask.size, U_mask.size))
ax1.plot(coords_meas_r_m_x, coords_meas_r_m_y, 'tab:blue')
ax1.plot(coords_meas_r_m_x.T, coords_meas_r_m_y.T, 'tab:blue')

coords_meas = transform(coords_mask, parameters_3_NLS)
ax1.plot(coords_meas[0,:],coords_meas[1,:], '+')
coords_meas_m_x = np.reshape(coords_meas[0,:],(U_mask.size, U_mask.size))
coords_meas_m_y = np.reshape(coords_meas[1,:],(U_mask.size, U_mask.size))
ax1.plot(coords_meas_m_x, coords_meas_m_y, 'tab:orange')
ax1.plot(coords_meas_m_x.T, coords_meas_m_y.T, 'tab:orange')

error1 = np.sum((coords_meas_r[1,:] - coords_meas[1,:])**2 + (coords_meas_r[0,:] - coords_meas[0,:])**2)
print("3_NLS error from coords")
print(error1/coords_meas[1,:].size)

coords_meas = transform(coords_mask, parameters_LS_NLS)
ax1.plot(coords_meas[0,:],coords_meas[1,:], 'x')
coords_meas_m_x = np.reshape(coords_meas[0,:],(U_mask.size, U_mask.size))
coords_meas_m_y = np.reshape(coords_meas[1,:],(U_mask.size, U_mask.size))
ax1.plot(coords_meas_m_x, coords_meas_m_y, 'tab:green')
ax1.plot(coords_meas_m_x.T, coords_meas_m_y.T, 'tab:green')

error2 = np.sum((coords_meas_r[1,:] - coords_meas[1,:])**2 + (coords_meas_r[0,:] - coords_meas[0,:])**2)
print("LS_NLS error from coords")
print(error2/coords_meas[1,:].size)
print("their ratio")
print(error1/error2)

plt.show()

