import sympy as sym
from sympy import pprint
import math as m
import numpy as np
import scipy.optimize
from sympy import pprint
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.optimize import minimize


w = m.e**((2/3)*m.pi*(1j))     # third root of unity
POVM_unnormalized = np.array([[0,1,-1],[-1,0,1],[1,-1,0],[0,w,-w**2],[-1,0,w**2],[1,-w,0],[0,w**2,-w],[-1,0,w],[1,-w**2,0]])  # unnormalized POVM direction vectors
POVM_vec = (1/(2**.5))*(np.array([[0,1,-1],[-1,0,1],[1,-1,0],[0,w,-w**2],[-1,0,w**2],[1,-w,0],[0,w**2,-w],[-1,0,w],[1,-w**2,0]]))  # normalized POVM direction vectors


"""def qutrit_dot(qutrit1, qutrit2):
    q1 = np.array(qutrit1)
    q2 = np.array(qutrit2)
    return np.vdot(q1,q2) """



class POVM_relations:
    def __init__(self, POVM_vec_list):
        self.POVM_vec_list = POVM_vec_list
        self.inner_products = {}
    
    def dot_product(self, i, j):
        return np.vdot(self.POVM_vec_list[i], self.POVM_vec_list[j])

    @property
    def all(self):
        if len(self.POVM_vec_list) != 9:
            raise ValueError('The POVM list must have 9 elements')
        for i in range(9):
            for j in range(i, 9):
                key = f'{i+1}{j+1}'
                value = self.dot_product(i, j)
                self.inner_products[key] = value
                # if abs(value) < 1e-18:
                    # self.inner_products[key] = 0
        return self.inner_products


# Another way, Creating a dictionary of pairs of Vector numbers and their inner product
# POVM_vec_np = (1/(2**0.5)) * np.array(POVM_unnormalized)
POVM_vec_np = np.array(POVM_unnormalized)
def qm_inner_product(vec1, vec2):
    return np.vdot(vec1, vec2)
inner_products = {}             # Compute the inner products and store in a dictionary
for i in range(len(POVM_vec_np)):
    for j in range(len(POVM_vec_np)):
        key = (i+1, j+1)
        value = qm_inner_product(POVM_vec_np[i], POVM_vec_np[j])
        if value.imag < 1e-14: 
            value = value.real 
            # value = round(value, 8)
        inner_products[key] = value 
# for key, value in inner_products.items():
#     print(f"Inner product of vectors {key}: {value}")
# verified it is symmertric, normalization holds etc.


#now finding the fifth element each by using the orthogonality condition of vector number 2 (which is known) and vectors j (number 3 and after)

c4j_list = [2,.5,.5,.5, (-.25-.433013j), -.25, .5, -.25, (-.25-.433013j)]           # fourth elements found earlier, not normalized at this point
# extending the 3d povm to 4d now that the fourth elements are known. this will be used for inner products and subsequently to find the fifth element.
four_lists = []                         # creating a list of the first four elements of each vector.
for i in range(9):
    # four_list = POVM_unnormalized[i].append(c4j_list[i])
    four_list = np.append(POVM_unnormalized[i], c4j_list[i])
    four_lists.append(four_list)


# storing possible pairs' inner products
four_inner_products = {}             
for i in range(len(four_lists)):
    for j in range(len(four_lists)):
        key = (i+1, j+1)
        value = np.vdot(four_lists[i], four_lists[j])
        if value.imag < 1e-14: 
            value = value.real 
            # value = round(value, 8)
        four_inner_products[key] = value

c5j_list = []                   # storing the fifth element for each vector after computing it.
for j in range(1,10):
    pair = (2,j)
    # print(f'four-inner product value associated to four_vectors: {pair} : ', four_inner_products[pair])
    c5j = -(2/np.sqrt(15))*four_inner_products[pair]           # see notebook
    # print(f'c5{pair[1]}:', c5j)
    c5j_list.append(c5j)

# print('list of 5th entries (c52 is wrong, should be sqrt(15)/2): ',c5j_list)        # second entry is wrong here, careful

#correct list, Second entry corrected from -1.1618 to root(15)/2 (see notebook)
c5j_list = [-0.0, np.sqrt(15)/2 , 0.38729833462074165, (-0.38729833462074176-0.44721359549995776j), -0.19364916731037082, 0.5809475019311124, -0.38729833462074165, (-0.19364916731037093-0.4472135954999579j), 0.5809475019311124]


# creating List of first five entries, Used for creating full vector later
five_lists = []                         
for i in range(9):
    five_list = np.append(four_lists[i], c5j_list[i])
    five_lists.append(five_list)

# inner products of the 5d vectors, for normalization 
five_inner_products = {}
for i in range(len(five_lists)):
    key = (i+1, i+1)
    value = np.vdot(five_lists[i], five_lists[i])
    if value.imag < 1e-14: 
        value = value.real 
        # value = round(value, 12)
    five_inner_products[key] = value


# print('Inner products of 5d vectors: ', five_inner_products)         


symbols = sym.symbols('c61, c71, c81, c91, c62, c72, c82, c92, c63, c73, c83, c93, c64, c74, c84, c94, c65, c75, c85, c95, c66, c76, c86, c96, c67, c77, c87, c97, c68, c78, c88, c98, c69, c79, c89, c99')


# Create the full 10D vectors v_i. Context: there are 9 free variables, we are setting the 9th enttry to be identically equal to 0 (tenth one is already zero). 
# There is another, probably better, method of cleverly using these 9 variables to actually reduce the number of equations. In that case, we use these 9 variables in the first 2 vectors, 5 and 4 
# respectively and then using these vectors we eliminate further coefficients or the unknown variables leaving us with actually only 21 equations(or something like that) .
NineDimVs_initial = []
for i in range(9):
    known_part = five_lists[i]            # Numerical entries from the five lists
    abstract_part = symbols[i*4:(i+1)*4]                         #  symbolic entries
    vector =  (1/sym.sqrt(6))*sym.Matrix(np.append(known_part, abstract_part)) 
    # NineDimVs.append(vector)
    NineDimVs_initial.append(vector.evalf())

NineDimVs_initial[0] = NineDimVs_initial[0].subs({'c61':0,'c71':0,'c81':0,'c91':0})     #manually changing the first vector and the second vector (The unknowns were all set to zero).
NineDimVs_initial[1] = NineDimVs_initial[1].subs({'c62':0,'c72':0,'c82':0,'c92':0})

u1_4, u2_4, u3_4, u4_4, u5_4, u6_4, u7_4, u8_4, u9_4 = sym.Matrix(NineDimVs_initial[0]),sym.Matrix(NineDimVs_initial[1]),sym.Matrix(NineDimVs_initial[2]),sym.Matrix(NineDimVs_initial[3]),sym.Matrix(NineDimVs_initial[4]),sym.Matrix(NineDimVs_initial[5]),sym.Matrix(NineDimVs_initial[6]),sym.Matrix(NineDimVs_initial[7]),sym.Matrix(NineDimVs_initial[8])
#above u vectors are the normalized vectors. the '_4' is Due to the fact that there are four unknown entries in each of those vectors (except the first two vectors which are fully known).

# pprint(np.sqrt(6)*u3_4) #eg (visualizing the unnormalized one, coz they are easier to understand)
# pprint(sym.sqrt(6)*u4)


# creating equations now amongst the last seven equations, all possible pairs, giving total 21 equations
# Not forgetting to take conjugates while doing inner product.

# Assuming NineDimVecs is a list of NumPy arrays, if not, convert them first
# For example, if originally in SymPy: u1 = np.array([complex(sym.re(u1), sym.im(u1)) for u1 in sympy_vector])

# List of nine vectors
# NineDimVecs = [u1, u2, u3, u4, u5, u6, u7, u8, u9]                   # uncomment this line and comment the next one if using the last entry as normalization entry
NineDimVecs = [u1_4, u2_4, u3_4,u4_4,u5_4,u6_4,u7_4,u8_4,u9_4]      
Equations_ortho_dict = {}

# Create pairs of equations and store in dictionary
for i in range(len(NineDimVecs)):
    for j in range(i , len(NineDimVecs)):
        key = f'E_{i+1}{j+1}'  # Create key as 'E_ij'
        # Compute inner product using NumPy with conjugate
        expression = np.vdot(NineDimVecs[i], NineDimVecs[j])
        Equations_ortho_dict[key] = expression

# how to substitute values in the equations
# tt = Equations_ortho_dict['E_35']
# pprint(tt)
# # print('','\n \n \n \n \n \n ',)
# print(tt.subs({'c64':0,'c74':0,'c84':0, 'c63':0, 'c73':0, 'c83':0, 'c65': 0,'c75': 0, 'c85': 0}))
# print(Equations_ortho_dict)

# lambdifying for faster numerical evaluation. Below is lambdifyied dictionary of equations
## uncomment the below line and comment out the line afterwards in order to use the version with normalization entry
# equations_lambdas = {key: sym.lambdify(('c63', 'c73', 'c83', 'c64', 'c74', 'c84', 'c65', 'c75', 'c85', 'c66', 'c76', 'c86', 'c67', 'c77', 'c87', 'c68', 'c78', 'c88', 'c69', 'c79', 'c89'), eq, 'numpy') for key, eq in Equations_ortho_dict.items()}
equations_lambdas = {key: sym.lambdify(('c63', 'c73', 'c83', 'c93', 'c64', 'c74', 'c84', 'c94', 'c65', 'c75', 'c85', 'c95', 'c66', 'c76', 'c86', 'c96', 'c67', 'c77', 'c87', 'c97', 'c68', 'c78', 'c88', 'c98', 'c69', 'c79', 'c89', 'c99'), eq, 'numpy') for key, eq in Equations_ortho_dict.items()}

# equations_lambdas['E_34'](1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)  # works

def Equations_ortho_obj_fn(vars):                                # vars is the list of variables (42 in total (56 if normalization entry is not in terms of other entries), first 21 are magnitudes, remaining 21 are corresponding phases)
    # if len(vars) < 42:  # in case less than 42 variables are given, we will pad it with zeros
    #     while len(vars) < 42:
    #         vars.append(0)
    
    # throw error if length of vars  is not even 
    if len(vars) % 2 != 0:
        raise ValueError('Number of variables must be even, check the input vars')

    magnitudes = vars[:int(len(vars)/2)]
    phases = vars[int(len(vars)/2):]
    complex_vars = [r * np.exp(1j * theta) for r, theta in zip(magnitudes, phases)]
    
    # Unpack complex variables as needed for your equations
    # c63, c73, c83, c64, c74, c84, c65, c75, c85, c66, c76, c86, c67, c77, c87, c68, c78, c88, c69, c79, c89 = complex_vars 
    c63, c73, c83, c93, c64, c74, c84, c94, c65, c75, c85, c95, c66, c76, c86, c96, c67, c77, c87, c97, c68, c78, c88, c98, c69, c79, c89, c99 = complex_vars 

    # Define or reference your equations here; this is a placeholder
    eqn_order = ['E_34', 'E_35', 'E_36', 'E_37', 'E_38', 'E_39', 'E_45', 'E_46', 'E_47', 'E_48', 'E_49', 'E_56', 'E_57', 'E_58', 'E_59', 'E_67', 'E_68', 'E_69', 'E_78', 'E_79', 'E_89']
    
    # Assuming Equations_ortho_dict is defined globally and contains the equations
    substituted_eqns = []
    for i in eqn_order:
        substituted_eqns.append(equations_lambdas[i](*complex_vars))
    
    residual_eqns = [abs(i)**2 for i in substituted_eqns]   # avoiding the problem of making both the real and imaginary parts zero individually. It works as long as the magnitudes/"residue" go to zero.
    output_residue = residual_eqns + residual_eqns  # for tricking the fsolve function to not throw errror "input length must match the output number of equations..". Doing this simply doubles the total residue.
    # return output_residue                         # , uncomment it and comment next when using fsolve instead of minimize
    return sum(output_residue)










# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************
# **************************The solving part************************************


# Creating a list of delta_ii, ie. the inner products of the five vectors with themselves. To be used in the normalization equations in the solver.
delta_five = []
for i in range(9):
    delta_five.append(np.vdot(five_lists[i], five_lists[i]))
print(delta_five)     # each elt is real, as it should be



# Define the function for the system of equations
def system_of_equations(vars):
    # Unpacking variables, C63 to C99 and y3 to y9
    C63, C64, C65, C66, C67, C68, C69, \
    C73, C74, C75, C76, C77, C78, C79, \
    C83, C84, C85, C86, C87, C88, C89, \
    C93, C94, C95, C96, C97, C98, C99, \
    y3, y4, y5, y6, y7, y8, y9 = vars
    global delta_five
    delta_11, delta_22, delta_33, delta_44, delta_55, delta_66, delta_77, delta_88, delta_99 = delta_five

    # Full set of 28 equations for Cij terms, based on previous interactions
    equations = [
        # Derivatives with respect to C63 to C69
        np.conj(C63)*y3,
        np.conj(C63) + np.conj(C64)*y4,
        np.conj(C63) + np.conj(C64) + np.conj(C65)*y5,
        np.conj(C63) + np.conj(C64) + np.conj(C65) + np.conj(C66)*y6,
        np.conj(C63) + np.conj(C64) + np.conj(C65) + np.conj(C66) + np.conj(C67)*y7,
        np.conj(C63) + np.conj(C64) + np.conj(C65) + np.conj(C66) + np.conj(C67) + np.conj(C68)*y8,
        np.conj(C63) + np.conj(C64) + np.conj(C65) + np.conj(C66) + np.conj(C67) + np.conj(C68) + np.conj(C69)*y9,
        # Continue with C73 to C79
        
        # Derivatives with respect to C73 to C79
        np.conj(C73)*y3,
        np.conj(C73) + np.conj(C74)*y4,
        np.conj(C73) + np.conj(C74) + np.conj(C75)*y5,
        np.conj(C73) + np.conj(C74) + np.conj(C75) + np.conj(C76)*y6,
        np.conj(C73) + np.conj(C74) + np.conj(C75) + np.conj(C76) + np.conj(C77)*y7,
        np.conj(C73) + np.conj(C74) + np.conj(C75) + np.conj(C76) + np.conj(C77) + np.conj(C78)*y8,
        np.conj(C73) + np.conj(C74) + np.conj(C75) + np.conj(C76) + np.conj(C77) + np.conj(C78) + np.conj(C79)*y9,

        # Derivatives with respect to C83 to C89
        np.conj(C83)*y3,
        np.conj(C83) + np.conj(C84)*y4,
        np.conj(C83) + np.conj(C84) + np.conj(C85)*y5,
        np.conj(C83) + np.conj(C84) + np.conj(C85) + np.conj(C86)*y6,
        np.conj(C83) + np.conj(C84) + np.conj(C85) + np.conj(C86) + np.conj(C87)*y7,
        np.conj(C83) + np.conj(C84) + np.conj(C85) + np.conj(C86) + np.conj(C87) + np.conj(C88)*y8,
        np.conj(C83) + np.conj(C84) + np.conj(C85) + np.conj(C86) + np.conj(C87) + np.conj(C88) + np.conj(C89)*y9,
        # Derivatives with respect to C93 to C99
        np.conj(C93)*y3,
        np.conj(C93) + np.conj(C94)*y4,
        np.conj(C93) + np.conj(C94) + np.conj(C95)*y5,
        np.conj(C93) + np.conj(C94) + np.conj(C95) + np.conj(C96)*y6,
        np.conj(C93) + np.conj(C94) + np.conj(C95) + np.conj(C96) + np.conj(C97)*y7,
        np.conj(C93) + np.conj(C94) + np.conj(C95) + np.conj(C96) + np.conj(C97) + np.conj(C98)*y8,
        np.conj(C93) + np.conj(C94) + np.conj(C95) + np.conj(C96) + np.conj(C97) + np.conj(C98) + np.conj(C99)*y9,
        #Now the normalization equations
        abs(C63)**2 + abs(C73)**2 + abs(C83)**2 + abs(C93)**2 + delta_33 - 1,
        abs(C64)**2 + abs(C74)**2 + abs(C84)**2 + abs(C94)**2 + delta_44 - 1,
        abs(C65)**2 + abs(C75)**2 + abs(C85)**2 + abs(C95)**2 + delta_55 - 1,
        abs(C66)**2 + abs(C76)**2 + abs(C86)**2 + abs(C96)**2 + delta_66 - 1,
        abs(C67)**2 + abs(C77)**2 + abs(C87)**2 + abs(C97)**2 + delta_77 - 1,
        abs(C68)**2 + abs(C78)**2 + abs(C88)**2 + abs(C98)**2 + delta_88 - 1,
        abs(C69)**2 + abs(C79)**2 + abs(C89)**2 + abs(C99)**2 + delta_99 - 1
    ]

    # Example placeholder for constraints based on previous description
    # equations.extend([
    #     # Example for y3
    #     y3 - (np.abs(C63)**2 + ...),  # Complete this based on your specific constraint
    #     # Continue with y4 to y9
    # ])
    # Ensure there are 35 equations in total
    assert len(equations) == 35
    return equations

# Initial guess for the variables (Cij real and imaginary parts, y)
# initial_guess = [0.1] * 35  
initial_guess = [0.2] * 15 + [-.2]*15 + [0.1] * 5  

# Solve the system of equations
solution = fsolve(system_of_equations, initial_guess)

# Print the solution
# print("Solution to the system:", solution)

# residues sum
residuals = system_of_equations(solution)
residuals_sum = np.sum(np.abs(residuals))
# print("Residuals:", residuals)
# print("Residuals sum:", residuals_sum)