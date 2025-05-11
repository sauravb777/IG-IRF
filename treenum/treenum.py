import math

def compute_accuracy(strength, correlation):
    return strength - correlation

def compute_qu_qv(u, v, f):
    def r(u,v,f): return 0 if v<f else math.comb(v,f)/math.comb(u+v,f)
    r0 = r(u,v,f); ru = r(u+1,v,f); rv = r(u,v+1,f)
    return -(ru-r0), -(rv-r0)

def compute_nu(q, rho, Nav, B):
    t1 = ((1-rho)**(B/2))/2 * math.log(1-rho) if rho<1 else 0
    t2 = (1-q**Nav)**B * math.log(1-q**Nav) if q**Nav<1 else 0
    return t1 - t2

def compute_l(q, Nav, B):
    return B * Nav * q**(Nav-1) * (1-q**Nav)**(B-1)

def compute_deltaB(qu, qv, du, dv, l, nu):
    num = l*(qu*du + qv*dv)
    return max(0, math.floor(abs(num/nu))) if nu!=0 else 0
