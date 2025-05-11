from math import comb

def compute_r(u, v, f):
    return 0 if v<f else comb(v, f) / comb(u+v, f)

def compute_q(u, v, f):
    return 1 - compute_r(u, v, f)

def compute_strength(q, Nav, B):
    return 1 - (1 - q**Nav)**B

def compute_correlation(u, v, f, Nav, B):
    total = u+v
    if total < 2*f:
        rho = 1.0
        return 1 - (1 - rho)**(B//2), rho
    rho_p = 1 - comb(total - f, f)/comb(total, f)
    rho   = rho_p**Nav
    return 1 - (1 - rho)**(B//2), rho
