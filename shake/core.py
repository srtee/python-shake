import numpy as np

class Constraints:
    def __init__(self,
                 npart=3,
                 ncons=2,
                 masses=np.array([1., 1., 1.]),
                 conlens=np.array([1., 1.]),
                 cons_matrix=np.array([[-1., 1., 0.], [-1., 0., 1.]])):
        self.npart = npart
        self.ncons = ncons
        self.masses = np.array(masses)
        self.L = np.atleast_2d(cons_matrix) # in the funny case of only one constraint?
        self.W = np.diag(1/self.masses) # needed for applying SHAKE
        self.conlens = conlens
        self.s_loaded = False
        self.r_loaded = False
        self.J_ilvesf = None
        if (self.consistency_check()):
            self.is_consistent = True
            self.K = self.L@self.W@self.L.T
        else:
            self.warn()
    
    def warn(self):
        print(f"""
        The provided masses (shape: {np.shape(self.masses)}), constraint lengths (shape: {np.shape(self.conlens)}),
        and constraint matrix (shape: {np.shape(self.L)}) are not consistent with
        the number of particles ({self.npart}) or constraints! ({self.ncons})""")
            
    def consistency_check(self):
        return (np.shape(self.masses) == (self.npart,) and
                np.shape(self.conlens) == (self.ncons,) and
                np.shape(self.L) == (self.ncons, self.npart))
    
    def load_s(self, s_coords=None):
        if (s_coords is None):
            if (not self.s_loaded):
                print("No s-coordinates found!")
                return 1
            else:
                return 0
        if (self.is_consistent):
            self.s_coords = np.array(s_coords)
            self.s_cons = self.L@self.s_coords
            self.s_conlensq = np.array([l@l for l in self.s_cons])
            self.s_loaded = True
            if (self.r_loaded):
                self.calculate_J()
            return 0
        else:
            self.warn()
            return 1
                
    def load_r(self, r_coords):
        if (r_coords is None):
            if (not self.r_loaded):
                print("No r-coordinates found!")
                return 1
            else:
                return 0
        if (self.is_consistent):
            self.r_coords = np.array(r_coords)
            self.r_cons = self.L@self.r_coords
            rr = self.r_cons@self.r_cons.T
            self.H = np.array([np.outer(k, k)*rr for k in self.K])
            self.J_ilvesf = self.K*rr
            self.r_loaded = True
            if (self.s_loaded):
                self.calculate_J()
            return 0
        else:
            self.warn()
            return 1
            
    def calculate_J(self):
        self.J = self.K*(self.s_cons@self.r_cons.T)
        self.Jinv = np.linalg.inv(self.J)
        
    def calculate_conlens(self, coords):
        if (self.is_consistent):
            return np.sqrt(np.array([l@l for l in self.L@coords]))
        else:
            self.warn()
                
    def shake(self, s_coords=None, r_coords=None, gamma=None, gammadiff=None):
        """does a single iteration of CANDLE aka matrix-based SHAKE """
        if (self.load_s(s_coords)):
            return 1
        if (self.load_r(r_coords)):
            return 1
        if gammadiff is None:
            gammadiff = np.zeros(self.ncons)
        if gamma is None:
            return 0.5*self.Jinv@(self.s_conlensq-self.conlens*self.conlens)
        else:
            gammasum = 2*gamma-gammadiff
            return 0.5*self.Jinv@(gammasum@self.H)@gammadiff
    
    def applyshake(self, s_coords=None, r_coords=None, gamma=None):
        """applies a gamma correction to the coordinates

        really we should be calling this applygamma????
        """
        if (gamma is None):
            gamma = self.shake(s_coords, r_coords)
        if (s_coords is None and self.s_loaded):
            s_coords = self.s_coords
        if (r_coords is None and self.r_loaded):
            r_coords = self.r_coords
        if (self.is_consistent):
            return (s_coords-self.W@self.L.T@np.diag(gamma)@self.L@r_coords).copy()
        else:
            self.warn()
    
    def itershake(self, s_coords=None, r_coords=None, maxsteps=50, thresh=1e-6):
        """iterates "CANDLE" until convergence
        
        """
        step = 0
        gamma = self.shake(s_coords, r_coords)
        gammadiff = gamma.copy()
        while step < maxsteps and np.max(np.fabs(gammadiff)) > thresh:
            yield gamma.copy()
            gammadiff = self.shake(s_coords, r_coords, gamma, gammadiff)
            gamma += gammadiff
            step += 1

    def ilvesm(self, s_coords=None, r_coords=None, gamma=None, gammadiff=None):
        """single step of ILVES-M

        """
        if (gamma is None):
            gamma = np.zeros(self.ncons)
        if (self.load_s(s_coords)):
            return 1 # by convention, if a computer program runs okay then it returns 0 if anything
        # this way, we can do "if (FUNCTION) {RUN ERROR HANDLER}"
        if (self.load_r(r_coords)):
            return 1
        
        ## calculate sigma
        new_s_coords = self.applyshake(gamma=gamma)
        # the first "gamma" is the keyword to the function -- the second "gamma" is the variable from earlier
        new_s_cons = self.L@new_s_coords
        new_sigma = np.array([l@l for l in new_s_cons]) - self.conlens*self.conlens
               
        ## calculate dsigma/dgamma
        dsigma_dgamma = -2*(self.K*(new_s_cons@self.r_cons.T))
        
        ## return new gamma
        new_gamma = gamma.copy() - np.linalg.inv(dsigma_dgamma)@new_sigma
        return new_gamma

    def iterilvesm(self, s_coords=None, r_coords=None, maxsteps=50, thresh=1e-6):
        """iterates ILVES-M until convergence
        
        """
        gamma = np.zeros(self.ncons)
        gammadiff = np.ones(self.ncons)
        step = 0
        while step < maxsteps and np.max(np.fabs(gammadiff)) > thresh:
            yield gamma.copy()
            new_gamma = self.ilvesm(s_coords, r_coords, gamma)
            gammadiff = new_gamma - gamma
            gamma = new_gamma
            step += 1

    def ilvesf(self, s_coords=None, r_coords=None, gamma=None, gammadiff=None):
        """single step of ILVES-F

        """
        if (gamma is None):
            gamma = np.zeros(self.ncons)
        if (self.load_s(s_coords)):
            return 1 # by convention, if a computer program runs okay then it returns 0 if anything
        # this way, we can do "if (FUNCTION) {RUN ERROR HANDLER}"
        if (self.load_r(r_coords)):
            return 1
        
        ## calculate sigma
        new_s_coords = self.applyshake(gamma=gamma)
        # the first "gamma" is the keyword to the function -- the second "gamma" is the variable from earlier
        new_s_cons = self.L@new_s_coords
        new_sigma = np.array([l@l for l in new_s_cons]) - self.conlens*self.conlens      
        # ILVES-F approximate dsigma/dgamma is calculated in load_r because it only depends on r
        
        ## return new gamma
        new_gamma = gamma.copy() + 0.5*np.linalg.inv(self.J_ilvesf)@new_sigma
        return new_gamma

    def iterilvesf(self, s_coords=None, r_coords=None, maxsteps=50, thresh=1e-6):
        """iterates ILVES-F until convergence
        
        """
        gamma = np.zeros(self.ncons)
        gammadiff = np.ones(self.ncons)
        step = 0
        while step < maxsteps and np.max(np.fabs(gammadiff)) > thresh:
            yield gamma.copy()
            new_gamma = self.ilvesf(s_coords, r_coords, gamma)
            gammadiff = new_gamma - gamma
            gamma = new_gamma
            step += 1
        

def gen_coord(bond1=1.0, bond2=1.0, angle=120):
    """Generate Coordinates That Match The Input Bond Lengths And Angle

    Keyword arguments:
    bond1 -- bond length 1 (default 1.0)
    bond2 -- bond length 2 (default 1.0)
    angle -- angle in degrees (default 120)
    """

    r0 = np.array([0.,0.,0.]) #when doing arrays must use square brackets
    r1 = np.array([bond1,0.,0.])
    a_rad = angle*np.pi/180
    r2 = np.array([bond2*np.cos(a_rad), bond2*np.sin(a_rad),0.])
    return r0,r1,r2

def analyse_coord(r0,r1,r2):
    """Analysis
    Keyword Arguments:
    r0 -- position of particle 0 (central particle)
    r1 and r2 -- positions of particles 1 and 2 do not matter, bond lengths will just be switched
    """
    bond1 = np.sqrt(np.dot(r1-r0,r1-r0))
    bond2 = np.sqrt(np.dot(r2-r0,r2-r0))
    mag_a = bond1
    mag_b = bond2
    a_dot_b = np.dot(r1-r0,r2-r0)
    theta = np.arccos(a_dot_b/(mag_a*mag_b))
    theta_degrees = theta*(180/np.pi)
    return bond1, bond2, theta_degrees #return -> provides what the output we want eg. we dont care about mag_b specifically

def shift_coord(r0,r1,r2,rng,std=0.1):
    """Shifted Coords
    
    Kewyword Arguments:
    r0 -- position of particle o (central particle)
    r1 and r2 -- position of particles 1 and 2
    std -- standard deviation of the shifts
    rng -- random number generator
    """
    
    r0 = r0.copy() + rng.standard_normal(3) * std #+= means take variable then add in vector and save as new version of variable
    r1 = r1.copy() + rng.standard_normal(3) * std
    r2 = r2.copy() + rng.standard_normal(3) * std
    return r0, r1, r2


def noniter_3p2c(cons):
    """Sets up noniterative solution for 3 particle 2 constraints Argument: constraint (no default) """
    # need to have: mu0, mu01, mu02 # interestingly these are the components of the K matrix!!
    # check that we have 3 particles and 2 constraints and
    # check if s and r are loaded because the vectors are
    # calculated from those
    if (cons.npart != 3 or cons.ncons != 2 or cons.s_loaded != True or cons.r_loaded != True):
        print("Constraints are not ready!")
        return
    
    mu0 = cons.K[0,1]
    mu01 = cons.K[0,0]
    mu02 = cons.K[1,1]
    Sigma = (mu0/mu01 + mu02/mu0)/2
    delta = (mu0/mu01 - mu02/mu0)/2 
  
    # s01 and s02 are in cons.s_cons[0] and [1]
    # r01 and r02 are in cons.r_cons[0] and [1]
    # also need d1 and d2:
    (s01, s02) = cons.s_cons
    (r01, r02) = cons.r_cons
    (d0, d1) = cons.s_conlensq-cons.conlens*cons.conlens
    r01_len = np.sqrt(r01@r01)
    r02_len = np.sqrt(r02@r02)
    costheta = r01@r02/(r01_len * r02_len)
    sintheta = np.sqrt(1 - costheta*costheta)
    d_cot_theta = delta*costheta/sintheta
    Delta = np.sqrt(d_cot_theta*d_cot_theta + Sigma*Sigma)
    U1 = Delta + d_cot_theta
    n = np.sqrt(U1*U1 + Sigma*Sigma)
    u1 = U1/n
    u2 = Sigma/n
    umat = np.array([[u1, -u2], [u2, u1]])
    lambdaplus = np.sqrt(1 + costheta)
    lambdaminus = np.sqrt(1 - costheta)
    ctilde = np.array([[1, 1], [-1, 1]])@np.array([[1/lambdaminus, 0], [0, 1/lambdaplus]])@umat/np.sqrt(2)
    cmat = np.array([[1/(mu01 * r01_len), 0], [0, 1/(mu0 * r02_len)]])@ctilde
    H1diag = cmat.T@cons.H[1]@cmat
    eigc1 = H1diag[0, 0]
    eigc2 = H1diag[1, 1]
    cplus = (eigc1 + eigc2)/2
    cminus = (eigc1 - eigc2)
    #return cmat
    jmat = np.array([[s01@r01/r01_len, s01@r02/r02_len], [(mu0/mu01)*s02@r01/r01_len,(mu02/mu0)*s02@r02/r02_len]])
    jcmat = jmat@ctilde
    #cplus = (delta/sintheta)*(delta/sintheta) + Delta*Delta
    #cminus = 2*delta*Delta/sintheta
    j00 = jcmat[0, 0]
    j01 = jcmat[0, 1]
    (dprime1, jprime10, jprime11) = (np.array([d1, jcmat[1, 0], jcmat[1, 1]]) - eigc2*np.array([d0, jcmat[0, 0], jcmat[0, 1]]))/cminus
    R2 = jcmat[0, 0]*jcmat[0, 0] + jcmat[0,1]*jcmat[0,1] - d0
    D = jprime10*jprime10 - jprime11*jprime11 - dprime1
    A = (jprime10-j00)/jprime11
    B = ((jprime10-j00)*(jprime10-j00) - jprime10*jprime10 + dprime1)/(2*jprime11) - j01
    return (A, B, R2, jprime11, cmat, jcmat)

from scipy.optimize import fsolve, root

def noniter_3p2c_gamma(cons, guess=None, threshold=1e-6):
    """Calculates gamma for 3 particle 2 constraints
    
    Arguments:
    
    constraint (no default),
    guess -- if None (default), uses initial guess + 1 Newton refinement
    threshold (default 1e-12) -- stop and return gamma when guess changes less than this
    
    """

    ## Solve quartic X^2 + (X^2/(2 jprime11) - A*X + B)^2 = R2

    A, B, R2, jprime11, cmat, jcmat = noniter_3p2c(cons)
    print(R2)

    def Y(x):
      return x**2/(2*jprime11) - A*x + B
    def quartic(x):
      return x**2 + Y(x)**2 - R2
    def derivative_of_quartic(x):
      return 2*x + 2*Y(x)*(x/jprime11 - A)
    if guess is None:
      """
      Best non-iterative initial guess for X based on SimDiagParab Eq. (11).

      Case 1: discriminant >= 0
        - Use the ± formula from Eq. (11) to get two candidates.
        - Evaluate the quartic f = X^2 + Y^2 - R^2 at both and keep
          the one with smaller |f|.

      Case 2: discriminant < 0
        - The line–circle approximation has no real intersection.
        - Try X = -R and X = +R and keep the one with smaller |f|.
      """

    # discriminant from SimDiagParab Eq(11)
      disc = A * A * R2 + R2 - B * B
      R = np.sqrt(R2)
      denom = A * A + 1.0

    # Case 1: discriminant >= 0 -> use plus minus branch from Eq(11)
      if disc >= 0.0:
        sqrt_disc = np.sqrt(disc)
        X_plus  = (A * B + sqrt_disc) / denom
        X_minus = (A * B - sqrt_disc) / denom
    # Case 2: discriminant < 0 -> fall back to plus minus R and choose best 
      else:
        X_plus = R
        X_minus = -R

    # choose the branch with smaller |quartic|
      print(f"quartic of X_plus {X_plus} is {quartic(X_plus)}")
      print(f"quartic of X_minus {X_minus} is {quartic(X_minus)}")
      if abs(X_plus) < abs(X_minus):
        guess = X_plus
      else:
        guess = X_minus

      # one step Newton refinement
      deriv = derivative_of_quartic(guess)
      #guess = guess
      if deriv != 0.0:
        guess = guess - quartic(guess)/deriv
    j00 = jcmat[0][0]
    roots = root(quartic, -j00, jac = derivative_of_quartic)
    X = roots.x[0]
    Yval = Y(X)
    j00 = jcmat[0][0]
    j01 = jcmat[0][1]
    gamma = cmat@np.array([X + j00, Yval + j01])

    q = quartic(X)
    print("Initial guess X0:", guess)
    print("Converged X:", X)
    print("Quartic value at X:", q)
    print("Root success flag:", roots.success)
    print("Function evaluations:", roots.nfev, "Jacobian evaluations:", roots.njev)

    return gamma
