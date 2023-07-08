import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

N=100
rho_w = 0.4
rho_s = rho_w+0.0001


def process(N, n, theta, rho):
    return theta*(((1-rho)*theta+rho)**(n-1))*(((1-rho)*(1-theta))**(N-n))+(1-theta)*((1-(1-rho)*theta)**(n-1))*(((1-rho)*theta)**(N-n))
    # algebra has been done incorrectly
    # return theta*(1+rho*thetac)**(n-1)*(thetac-rho*thetac)**(N-n)+thetac*(thetac+rho*theta)**(n-1)*(theta-rho*theta)**(N-n)

def result(N, n, theta, rho_w, rho_s):
    return (1+(process(N,n, theta, rho_w)/process(N,n,theta, rho_s)))**-1

def g(N,n,theta,rho):
    thetac=1-theta
    rhoc = 1-rho
    return theta*(theta+rho*thetac)**(n-1) * (rhoc*thetac)**(N-n)

def alpha(N,n,theta,rho):
    return g(N,n,theta,rho)/g(N,n,1-theta,rho)

def prefix(theta,rho):
    return theta*(theta+rho*(1-theta))

def Delta(N,n,theta,rho_s,rho_w):
    thetac = 1-theta
    rhoc_s = 1-rho_s
    rhoc_w = 1-rho_w
    frag_1 = rhoc_s/rhoc_w
    frag_2 = (2+alpha(N,n,theta,rho_s)+(1/alpha(N,n,theta,rho_s)))/(2+alpha(N,n,theta,rho_w)+(1/alpha(N,n,theta,rho_w)))
    frag_3_num = (prefix(theta, rho_w)*(1+alpha(N,n,theta,rho_w))+prefix(1-theta, rho_w)*(1+1/alpha(N,n,theta,rho_w)))
    frag_3_den = (prefix(theta, rho_s)*(1+alpha(N,n,theta,rho_s))+prefix(1-theta, rho_s)*(1+1/alpha(N,n,theta,rho_s)))
    return frag_1*frag_2*(frag_3_num/frag_3_den)

df = pd.DataFrame()
thetas = []
for theta in range(50,100):
    theta = theta/100
    thetas.append(theta)
    temp = []
    for n in range(N+1):
        temp.append(result(N, n, theta, rho_w, rho_s))
    df[f'theta={theta}'] = temp

df.index.name='n'

delta = df.diff()

points_pos = []
points_neg = []
for n in range(1,N+1):
    for theta in range(0,50):
        if delta.iloc[n,theta]>=0:
            points_pos.append([n,theta/100+0.5])
        else:
            points_neg.append([n,theta/100+0.5])

df_pos = pd.DataFrame(points_pos,columns=['n','theta'])
plt.scatter(df_pos['n'], df_pos['theta'], color= 'green', label="positive correlation")

df_neg = pd.DataFrame(points_neg,columns=['n','theta'])
plt.scatter(df_neg['n'], df_neg['theta'], color= 'red', label="negative correlation")

plt.legend(loc='upper left')
plt.title(f"dPr(rho_s|s_1=H,n)/dn for various theta and n where N={N}, rho_s={rho_s}, rho_w={rho_w}")
plt.xlabel("n")
plt.ylabel("theta")
plt.show()


df = pd.DataFrame()
thetas = []
for theta in range(50,100):
    theta = theta/100
    thetas.append(theta)
    temp = []
    for n in range(N):
        temp.append(Delta(N, n, theta, rho_w, rho_s))
    df[f'theta={theta}'] = temp

points_pos = []
points_neg = []
for n in range(1,N):
    for theta in range(0,50):
        if df.iloc[n,theta]>=1:
            points_pos.append([n,theta/100+0.5])
        else:
            points_neg.append([n,theta/100+0.5])

df_pos = pd.DataFrame(points_pos,columns=['n','theta'])
plt.scatter(df_pos['n'], df_pos['theta'], color= 'green', label="positive correlation")

df_neg = pd.DataFrame(points_neg,columns=['n','theta'])
plt.scatter(df_neg['n'], df_neg['theta'], color= 'red', label="negative correlation")

plt.legend(loc='upper left')
plt.title(f"dPr(rho_s|s_1=H,n)/dn for various theta and n where N={N}, rho_s={rho_s}, rho_w={rho_w}")
plt.xlabel("n")
plt.ylabel("theta")
plt.show()

## Test Delta
N=10
n=5
theta=0.7
rho_w = 0.4
rho_s = 0.6

D_1 = (process(N,n+1, theta, rho_w)/process(N,n+1,theta, rho_s))/(process(N,n, theta, rho_w)/process(N,n,theta, rho_s))
D_2 = Delta(N,n,theta,rho_s,rho_w)

D_1,D_2

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