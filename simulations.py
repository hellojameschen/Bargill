import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import comb

N=100
rho_w = 0.
rho_s = 0.001
theta = 0.5
intervals = 100

def pcc(theta, rho):
    return (1-rho)*theta+rho

def picc(theta, rho):
    return 1-pcc(theta,rho)

def pcic(theta, rho):
    return (1-rho)*theta

def picic(theta, rho):
    return 1-pcic(theta, rho)

def result(N,n,theta,rho):
    num = (1-theta)*n*picic(theta, rho)**(n-1)*pcic(theta,rho)**(N-n)+theta*(N-n)*picc(theta,rho)**n*pcc(theta,rho)**(N-n-1)
    den = theta *n*pcc(theta,rho)**(n-1) *picc(theta,rho)**(N-n)+ (1-theta)* (N-n)*pcic(theta,rho)**n* picic(theta,rho)**(N-n-1)
    return num/den

df = pd.DataFrame()
rhos = np.linspace(0,1,intervals,endpoint=False)
thetas = np.linspace(0,1,intervals,endpoint=False)
for rho in rhos:
    temp = []
    for n in range(N):
        temp.append(result(N, n, theta, rho))
    df[f'rho={rho}'] = temp

df.index.name='n'
df = df.T

delta = df.diff()

points_pos = []
points_neg = []
for n in range(1,N-1):
    for rho_idx in range(len(rhos)):
        if delta.iloc[n,rho_idx]>=0:
            points_pos.append([n,rhos[rho_idx]])
        else:
            points_neg.append([n,rhos[rho_idx]])

df_pos = pd.DataFrame(points_pos,columns=['n','rho'])
plt.scatter(df_pos['n']+1, df_pos['rho'], color= 'green', label="positive correlation")

df_neg = pd.DataFrame(points_neg,columns=['n','rho'])
plt.scatter(df_neg['n']+1, df_neg['rho'], color= 'red', label="negative correlation")

# plt.legend(loc='upper left')
plt.title(f"d(Pr(n|H)/Pr(H|n))/drho for various rho and n where N={N}, theta={theta}")
plt.xlabel("n")
plt.ylabel("rho")
plt.show()

# def process(N, n, theta, rho):
#     term1 = 0.5*comb(N-1,n-1)*(theta) * ((1-rho)*theta+rho)**(n-1) * ((1-rho)*(1-theta))**(N-n)
#     term2 = 0.5*comb(N-1,n)*(1-theta) * ((1-rho)*theta)**n * (1-(1-rho)*theta)**(N-n-1)
#     return term1+term2

# def process(N, n, theta, rho):
#     return theta*(((1-rho)*theta+rho)**(n-1))*(((1-rho)*(1-theta))**(N-n))+(1-theta)*((1-(1-rho)*theta)**(N-n))*(((1-rho)*theta)**(n-1))

# def process(N, n, theta, rho):
#     return theta*(((1-rho)*theta+rho)**(n-1))*(((1-rho)*(1-theta))**(N-n))+(1-theta)*((1-(1-rho)*theta)**(N-n))*(((1-rho)*theta)**(n-1))
    # algebra has been done incorrectly
    # return theta*(1+rho*thetac)**(n-1)*(thetac-rho*thetac)**(N-n)+thetac*(thetac+rho*theta)**(n-1)*(theta-rho*theta)**(N-n)

# def result(N, n, theta, rho_w, rho_s):
#     return (1+(process(N,n, theta, rho_w)/process(N,n,theta, rho_s)))**-1

# def g(N,n,theta,rho):
#     thetac=1-theta
#     rhoc = 1-rho
#     return theta*(theta+rho*thetac)**(n-1) * (rhoc*thetac)**(N-n)

# def alpha(N,n,theta,rho):
#     return g(N,n,theta,rho)/g(N,n,1-theta,rho)

# def prefix(theta,rho):
#     return theta*(theta+rho*(1-theta))

# def get_frag_1(N,n,theta,rho_w,rho_s):
#     rhoc_s = 1-rho_s
#     rhoc_w = 1-rho_w
#     return rhoc_s/rhoc_w

# def get_frag_2(N,n,theta,rho_w,rho_s):
#     thetac = 1-theta
#     rhoc_s = 1-rho_s
#     rhoc_w = 1-rho_w
#     return (2+alpha(N,n,theta,rho_s)+(1/alpha(N,n,theta,rho_s)))/(2+alpha(N,n,theta,rho_w)+(1/alpha(N,n,theta,rho_w)))

# def get_frag_3_half(N,n,theta,rho):
#     return (prefix(theta, rho)*(1+alpha(N,n,theta,rho))+prefix(1-theta, rho)*(1+1/alpha(N,n,theta,rho)))

# def Delta(N,n,theta,rho_w,rho_s):
#     frag_1 = get_frag_1(N,n,theta,rho_w,rho_s)
#     frag_2 = get_frag_2(N,n,theta,rho_w,rho_s)
#     frag_3_num = get_frag_3_half(N,n,theta,rho_w)
#     frag_3_den = get_frag_3_half(N,n,theta,rho_s)
#     return (frag_1*frag_2*(frag_3_num/frag_3_den))

# df = pd.DataFrame()
# thetas = []
# for theta in range(40,100):
#     theta = theta/100
#     thetas.append(theta)
#     temp = []
#     for n in range(1,N):
#         temp.append(result(N, n, theta, rho_w, rho_s))
#     df[f'theta={theta}'] = temp

# df.index.name='n'

# delta = df.diff()

# points_pos = []
# points_neg = []
# for n in range(1,N-1):
#     for theta in range(0,50):
#         if delta.iloc[n,theta]>=0:
#             points_pos.append([n,theta/100+0.5])
#         else:
#             points_neg.append([n,theta/100+0.5])

# df_pos = pd.DataFrame(points_pos,columns=['n','theta'])
# plt.scatter(df_pos['n']+1, df_pos['theta'], color= 'green', label="positive correlation")

# df_neg = pd.DataFrame(points_neg,columns=['n','theta'])
# plt.scatter(df_neg['n']+1, df_neg['theta'], color= 'red', label="negative correlation")

# plt.legend(loc='upper left')
# plt.title(f"dPr(rho_s|s_1=H,n)/dn for various theta and n where N={N}, rho_s={rho_s}, rho_w={rho_w}")
# plt.xlabel("n")
# plt.ylabel("theta")
# plt.show()


# df = pd.DataFrame()
# thetas = []
# for theta in range(50,100):
#     theta = theta/100
#     thetas.append(theta)
#     temp = []
#     for n in range(N):
#         temp.append(Delta(N, n, theta, rho_w, rho_s))
#     df[f'theta={theta}'] = temp

# points_pos = []
# points_neg = []
# for n in range(1,N):
#     for theta in range(0,50):
#         if df.iloc[n,theta]<=1:
#             points_pos.append([n,theta/100+0.5])
#         else:
#             points_neg.append([n,theta/100+0.5])

# df_pos = pd.DataFrame(points_pos,columns=['n','theta'])
# plt.scatter(df_pos['n'], df_pos['theta'], color= 'green', label="positive correlation")

# df_neg = pd.DataFrame(points_neg,columns=['n','theta'])
# plt.scatter(df_neg['n'], df_neg['theta'], color= 'red', label="negative correlation")

# plt.legend(loc='upper left')
# plt.title(f"dPr(rho_s|s_1=H,n)/dn for various theta and n where N={N}, rho_s={rho_s}, rho_w={rho_w}")
# plt.xlabel("n")
# plt.ylabel("theta")
# plt.show()

## Test Delta
# N=100
# # n=57
# n=70
# theta=0.6
# rho_w = 0.1
# rho_s = rho_w+0.00000000000001

# D_1 = (process(N,n+1, theta, rho_w)/process(N,n+1,theta, rho_s))/(process(N,n, theta, rho_w)/process(N,n,theta, rho_s))
# D_2 = Delta(N,n,theta,rho_w,rho_s)

# D_1,D_2, result(N,n+1, theta, rho_w, rho_s)/result(N,n, theta, rho_w, rho_s)

# tup = get_frag_1(N,n,theta,rho_w,rho_s),get_frag_2(N,n,theta,rho_w,rho_s),get_frag_3_half(N,n,theta,rho_w)/get_frag_3_half(N,n,theta,rho_s),D_2
# for num in tup:
#     print(num)

# D_2,get_frag_1(N,n,theta,rho_w,rho_s)*get_frag_2(N,n,theta,rho_w,rho_s)*get_frag_3_half(N,n,theta,rho_w)/get_frag_3_half(N,n,theta,rho_s)

# N=10
# rho_w = 0.1
# rho_s = 0.5
# n=7
# theta = 0.9
# result(N, n, theta, rho_w, rho_s)

# rhos = np.linspace(0,1,num=5, endpoint=False)
# N=10
# theta_vals = 10

# lst = []
# for rho in rhos:
#     df = pd.DataFrame()
#     thetas = []
#     for theta in range(theta_vals,theta_vals*2):
#         theta = theta/(2*theta_vals)
#         thetas.append(theta)
#         temp = []
#         for n in range(N+1):
#             temp.append(process(N, n, theta, rho))
#         df[f'{theta}'] = temp
#     lst.append(df)

# for i in range(theta_vals):
#     plt.plot(lst[0].iloc[:,i], label = lst[0].columns[i])

# plt.legend()
# plt.xlabel('n')

# for i in range(theta_vals)[1:]:
#     plt.plot(lst[0].iloc[i,:], label = f'n={i}')

# plt.legend()
# plt.xlabel('theta')


# theta=0.8
# N=100
# n=60
# rho_ws= np.linspace(0.1,0.9,50,endpoint=False)
# rho_ss= rho_ws+0.000001

# results = []

# for rho_w,rho_s in zip(rho_ws, rho_ss):
#     results.append(Delta(N, n, theta, rho_w, rho_s))

# plt.plot(rho_ws, results)


# for column in df.columns[::5]:
#     plt.plot(df.index+1,df[column], label=column)
# plt.xlabel('n')
# plt.ylabel('Pr(rho_s|s_1=H,n)')
# plt.legend()
# plt.title(f"Pr(rho_s|s_1=H,n) for various theta and n where N={N}, rho_s={rho_s}, rho_w={rho_w}")
# plt.show()


# cols=['theta=0.5']#,'theta=0.9']
# for col in cols:
#     plt.plot(df.index+1,df[col], label=col)
# # plt.axhline(0.5, label='0.5')
# plt.xlabel('n')
# plt.ylabel('Pr(rho_s|s_1=H,n)')
# plt.legend()
# plt.title(f"Pr(rho_s|s_1=H,n) for various n where theta=0.5 N={N}, rho_s={rho_s}, rho_w={rho_w}")

# plt.show()