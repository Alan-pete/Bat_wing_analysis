import numpy as np
from matplotlib import pyplot as plt
import json

class wing:

    def __init__(self, file_name):
        self.nodes = self.get_nodes(file_name)  # number of nodes
        self.theta = self.get_theta()  # theta values using cosine clustering
        self.zb = self.zb_function()  # z/b function values
        self.planformcb = self.get_planform(file_name)  # pulling in any planform and giving the chord distribution
        self.area = self.get_wing_area(file_name)  # calculates the area of the wing
        self.ra = self.get_ra(file_name)  # aspect ratio
        self.cb = self.cb_function(file_name)  # c/b function values
        self.w = self.get_w(file_name)  # washout distribution
        self.c_mat = self.get_cmatrix(file_name)  # c matrix
        self.c_inv = self.get_cinverse()  # c inverse matrix
        self.cfc = self.get_cfc(file_name)  # aileron distribution
        self.epsilon_fi = self.get_epsilon_fi()  # ideal flap effectiveness
        self.epsilon_f = self.get_epsilon_f(file_name)  # section flap effectiveness
        self.chi = self.get_chi(file_name)  # aileron distribution
        self.a_n = self.get_a_n()  # a Fourier coefficients
        self.b_n = self.get_b_n()  # b Fourier coefficients
        self.c_n = self.get_c_n()  # c Fourier coefficients
        self.d_n = self.get_d_n()  # d Fourier coefficients
        self.k_L = self.get_k_L(file_name)  # kappa L value
        self.k_D = self.get_k_D()  # kappa D value
        self.cl_alpha = self.get_cl_alpha(file_name)  # wing lift slope
        self.e_o = self.get_epsilon_o()  # getting epsilon omega value
        self.k_DL = self.get_k_DL(file_name)  # getting kappa DL value
        self.k_D_omega = self.get_k_d_omega(file_name)  # getting kappa D omega value
        self.k_lp = self.get_k_lp()  # roll damping factor
        self.c_ld = self.get_c_ld(file_name)  # deflection component of integrated rolling moment
        self.c_lp = self.get_c_lp()  # rolling rate component of integrated rolling moment
        self.p_bar = self.get_p_bar(file_name)  # rolling rate
        self.es = self.get_es()  # Span efficiency factor
        self.omega = self.get_omega(file_name)  # Omega amount
        self.alpha_root = self.get_alpha_root(file_name)  # alpha root degree when given CL
        self.A_n = self.get_A_n(file_name)  # final Fourier coefficients
        self.cl_roll = self.get_cl_roll(file_name)  # rolling-moment coefficient
        self.c_nyaw = self.get_cn_yaw(file_name)  # yawing-moment coefficient
        self.cl = self.get_cl(file_name)  # coefficient of lift for the wing
        self.cd_i = self.get_cD_i(file_name)  # coefficient of induced drag neglecting aileron and rolling rate
        self.cd_ia = self.get_cD_i_aileron(file_name)  # coefficient of induced drag neglecting aileron and rolling rate
        self.cl_hat_planform = self.get_cl_hat_planform()  # contribution to dimensionless lift distribution from planform
        self.cl_hat_washout = self.get_cl_hat_washout()  # contribution to dimensionless lift distribution from washout
        self.cl_hat_aileron = self.get_cl_hat_aileron(file_name)  # contribution to dimensionless lift distribution from ailerons
        self.cl_hat_roll = self.get_cl_hat_roll()  # contribution to dimensionless lift distribution from rolling rate
        self.cl_hat = self.get_cl_hat()  # total contribution to dimensionless lift distribution
        self.cl_tilde_planform = self.get_cl_tilde_planform()  # contribution to the local lift coefficient from planform
        self.cl_tilde_washout = self.get_cl_tilde_washout()  # contribution to the local lift coefficient from washout
        self.cl_tilde_aileron = self.get_cl_tilde_aileron()  # contribution to the local lift coefficient from aileron
        self.cl_tilde_roll = self.get_cl_tilde_roll()  # contribution to the local lift coefficient from aileron
        self.cl_tilde = self.get_cl_tilde()  # total contribution to the local lift coefficient
        self.write = self.write_to_file(file_name)  # writing the appropriate values to a txt file
        self.planform_dist = self.get_plot_planform(file_name)  # plot of the planform
        self.washout_dist = self.get_plot_washout(file_name)  # plot of the washout distribution
        self.aileron_dist = self.get_plot_aileron(file_name)  # plot of the aileron distribution
        self.clhat_dist = self.get_plot_cl_hat(file_name)  # plot of the cl_hat distribution
        self.cl_tilde_dist = self.get_plot_cl_tilde(file_name)  # plot of the cl_tilde distribution

    def get_nodes(self, file_name):
        # getting number of nodes for planform from json file
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        node = input_dict["wing"]
        n_half = node["nodes_per_semispan"]  # reads in nodes per semispan
        n = (2 * n_half) - 1
        return n

    def get_theta(self):
        # returns the value of theta for each node
        theta = np.linspace(0.0, np.pi, self.nodes)
        np.array(theta)
        return theta

    def zb_function(self):
        # returns the value of z/b for each node
        zb = -0.5 * np.cos(self.theta)
        np.array(zb)
        return zb

    def get_planform(self, file_name):
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        wing = input_dict["wing"]
        planform = wing["planform"]["type"]
        file = wing["planform"]["filename"]  # reads in the txt file specified by the json file

        if planform == "file":
            with open(file, "r") as f:
                info = []
                f.readline()  # skips the first line in the txt file
                for line in f:
                    info.append(
                        [float(coordinate) for coordinate in line.split()])  # splits the list up into x and y values
                info = np.array(info)
                self.zbfile = info[:, 0]
                self.cbfile = info[:, 1]
                cb = np.interp(abs(self.zb), self.zbfile, self.cbfile)
                return cb
        else:
            return planform

    def get_wing_area(self, file_name):
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        wing = input_dict["wing"]
        planform = wing["planform"]["type"]

        if planform == "file":
            area = np.trapz(self.cbfile, self.zbfile)
            return area

    def get_ra(self, file_name):
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        wing = input_dict["wing"]
        planform = wing["planform"]["type"]

        if planform == "file":
            ra = (2 * self.zb[-1]**2)/self.area
            return ra
        else:
            ra = wing["planform"]["aspect_ratio"]
            return ra

    def cb_function(self, file_name):
        # reads in the chord distribution depending on the planform, aspect ratio, and taper ratio
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        planform = shape["planform"]["type"]  # reads in the type of planform

        if planform == "elliptic":
            # r_a = shape["planform"]["aspect_ratio"]
            cb = (4/(np.pi * self.ra)) * np.sqrt(1 - (2 * self.zb)**2)  # chord distribution using z/b value
            # cb_theta = (4/(np.pi * r_a)) * np.sin(self.theta)  # chord distribution using theta value
            # making sure that no c/b value is zero
            for i in range(len(cb)):
                if abs(cb[i]) < 0.00001:
                    cb[i] = 0.00001
            np.array(cb)
            return cb

        if planform == "tapered":
            # r_a = shape["planform"]["aspect_ratio"]
            r_t = shape["planform"]["taper_ratio"]
            cb = (2/(self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(2 * self.zb)))  # chord distribution using z/b value
            # cb_theta = (2/(self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(np.cos(self.theta))))  # chord distribution using theta value
            for i in range(len(cb)):
                if abs(cb[i]) < 0.00001:
                    cb[i] = 0.00001
            np.array(cb)
            return cb

        if planform == "file":
            cb = self.planformcb
            return cb

    def get_w(self, file_name):
        # pulling in the washout
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        washout = shape["washout"]["distribution"]

        # washout distributions for different types of washout
        if washout == "none":
            w = np.zeros((self.nodes, 1))
            return w

        if washout == "linear":
            w = abs(2 * self.zb)  # linear washout using z/b value
            # w_theta = abs(np.cos(self.theta))  # linear washout using theta
            return w

        if washout == "optimum":
            b_3 = shape["washout"]["B3"]
            cl_alpha = shape["airfoil_lift_slope"]
            c_root = shape["nodes_per_semispan"] - 1
            sin = 0
            for i in range(len(self.theta)):
                sin = np.sin(self.theta[i])
                if sin < 0.00001:
                    sin = 0.00001
            w = ((4/cl_alpha) * (((1 - b_3)/self.cb[c_root]) - (np.sin(self.theta) + (b_3 * np.sin(3 * self.theta)))/self.cb) - (3 * b_3 * (1 + (np.sin(3 * self.theta)/sin)))) / (((4 * (1 - b_3))/(cl_alpha * self.cb[c_root])) - (12 * b_3))
            return w

    def get_cmatrix(self, file_name):
        # pulls in the lift slope of the wing
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        cl_alpha1 = shape["airfoil_lift_slope"]

        # getting the c matrix
        c_mat = np.zeros((self.nodes, self.nodes))  # gives a matrix of zeros the size of nodes x nodes
        for j in range(0, self.nodes):
            c_mat[0, j] = (j+1)**2  # first row
            for i in range(1, self.nodes-1):
                c_mat[i, j] = (4/(cl_alpha1 * self.cb[i]) + (j+1)/(np.sin(self.theta[i]))) * np.sin((j+1) * self.theta[i])
            c_mat[self.nodes-1, j] = ((-1)**j) * (j+1)**2  # last row
        return c_mat

    def get_cinverse(self):
        i_mat = np.eye(self.nodes, self.nodes)  # identity matrix
        c_inv = np.linalg.solve(self.c_mat, i_mat)  # solving for c inverse matrix
        return c_inv

    def get_cfc(self, file_name):
        # pulls in the aileron values
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        aileron = input_dict["wing"]
        zb1 = aileron["aileron"]["begin[z/b]"]
        zb2 = aileron["aileron"]["end[z/b]"]
        cfc1 = aileron["aileron"]["begin[cf/c]"]
        cfc2 = aileron["aileron"]["end[cf/c]"]

        # reads in the chord distribution depending on the planform, aspect ratio, and taper ratio
        planform = aileron["planform"]["type"]  # reads in the type of planform

        if planform == "elliptic":
            # r_a = aileron["planform"]["aspect_ratio"]
            cb1 = (4 / (np.pi * self.ra)) * np.sqrt(1 - (2 * zb1)** 2)  # chord distribution at z/b value
            cb2 = (4 / (np.pi * self.ra)) * np.sqrt(1 - (2 * zb2)** 2)
            y1 = cb1 * (-0.75 + cfc1)
            y2 = cb2 * (-0.75 + cfc2)
            slope1 = (y2 - y1)/(zb2 - zb1)
            b1 = y2 - (slope1 * zb2)
            cfc = np.zeros((self.nodes, 1))
            # finding the flap-chord fraction
            for i in range(0, self.nodes):
                if zb1 <= self.zb[i] <= zb2:
                    y = (slope1 * self.zb[i]) + b1
                    cfc[i] = (y/self.cb[i]) + 0.75
                elif -zb2 <= self.zb[i] <= -zb1:
                    y = (-slope1 * self.zb[i]) + b1
                    cfc[i] = (y/self.cb[i]) + 0.75
                else:
                    cfc[i] = 0
            return cfc

        if planform == "tapered":
            # r_a = aileron["planform"]["aspect_ratio"]
            r_t = aileron["planform"]["taper_ratio"]
            cb1 = (2 / (self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(2 * zb1)))  # chord distribution using z/b value
            cb2 = (2 / (self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(2 * zb2)))
            y1 = cb1 * (-0.75 + cfc1)
            y2 = cb2 * (-0.75 + cfc2)
            slope1 = (y2 - y1) / (zb2 - zb1)
            b1 = y2 - (slope1 * zb2)
            cfc = np.zeros((self.nodes, 1))
            # finding the flap-chord fraction
            for i in range(0, self.nodes):
                if zb1 <= self.zb[i] <= zb2:
                    y = (slope1 * self.zb[i]) + b1
                    cfc[i] = (y / self.cb[i]) + 0.75
                elif -zb2 <= self.zb[i] <= -zb1:
                    y = (-slope1 * self.zb[i]) + b1
                    cfc[i] = (y / self.cb[i]) + 0.75
                else:
                    cfc[i] = 0
            return cfc

        if planform == "file":
            # finding the chord length at the end of each aileron
            cb1 = np.interp(zb1, self.zbfile, self.cbfile)
            cb2 = np.interp(zb2, self.zbfile, self.cbfile)

            y1 = cb1 * (-0.75 + cfc1)
            y2 = cb2 * (-0.75 + cfc2)
            slope1 = (y2 - y1) / (zb2 - zb1)
            b1 = y2 - (slope1 * zb2)
            cfc = np.zeros((self.nodes, 1))
            # finding the flap-chord fraction
            for i in range(0, self.nodes):
                if zb1 <= self.zb[i] <= zb2:
                    y = (slope1 * self.zb[i]) + b1
                    cfc[i] = (y / self.cb[i]) + 0.75
                elif -zb2 <= self.zb[i] <= -zb1:
                    y = (-slope1 * self.zb[i]) + b1
                    cfc[i] = (y / self.cb[i]) + 0.75
                else:
                    cfc[i] = 0
            return cfc

    def get_epsilon_fi(self):
        # ideal flap effectiveness
        theta_f = np.arccos((2 * self.cfc) - 1)
        epsilon_fi = 1 - ((theta_f - np.sin(theta_f))/np.pi)
        return epsilon_fi

    def get_epsilon_f(self, file_name):
        # section flap effectiveness
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        aileron = input_dict["wing"]
        eta_h = aileron["aileron"]["hinge_efficiency"]  # hinge efficiency
        eta_d = 1  # deflection efficiency
        epsilon_f = eta_h * eta_d * self.epsilon_fi
        return epsilon_f

    def get_chi(self, file_name):
        # aileron distribution
        # pulls in the aileron values
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        aileron = input_dict["wing"]
        zb1 = aileron["aileron"]["begin[z/b]"]
        zb2 = aileron["aileron"]["end[z/b]"]

        chi = np.zeros((self.nodes, 1))
        for i in range(0, self.nodes):
            if self.zb[i] < -zb2:
                chi[i] = 0
            if -zb2 < self.zb[i] < -zb1:
                chi[i] = self.epsilon_f[i]
            if -zb1 < self.zb[i] < zb1:
                chi[i] = 0
            if zb1 < self.zb[i] < zb2:
                chi[i] = -self.epsilon_f[i]
            if self.zb[i] > zb2:
                chi[i] = 0
        return chi

    def get_a_n(self):
        # a coefficients
        b1 = np.ones((self.nodes, 1))  # column matrix of 1's
        a_n = np.matmul(self.c_inv, b1)  # multiplication to get a_n
        return a_n

    def get_b_n(self):
        # b coefficients
        b_n = np.matmul(self.c_inv, self.w)
        return b_n

    def get_c_n(self):
        # c coefficients
        c_n = np.matmul(self.c_inv, self.chi)
        return c_n

    def get_d_n(self):
        # d coefficients
        b4 = np.cos(self.theta)
        d_n = np.matmul(self.c_inv, b4)
        return d_n

    def get_k_lp(self):
        # roll damping factor
        k_lp = (2 * self.d_n[1]) / self.a_n[0]
        return k_lp

    def get_c_ld(self, file_name):
        # to find the deflection component of the integrated rolling moment
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        c_ld = -((np.pi * self.ra)/4) * self.c_n[1]
        print("Cl_da: ", c_ld)
        return c_ld

    def get_c_lp(self):
        # to find the rolling rate component of the integrated rolling moment
        c_lp = -(self.k_lp * self.cl_alpha)/8
        print("Cl_pbar: ", c_lp)
        return c_lp

    def get_p_bar(self, file_name):
        # pulling in the info for rolling rate
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        roll = input_dict["condition"]
        pbar = roll["pbar"]
        delta_a = np.radians(roll["aileron_deflection[deg]"])

        if pbar == "steady":
            p_bar = -(self.c_ld/self.c_lp) * delta_a
            print("p_bar: ", p_bar)
            return p_bar
        else:
            p_bar = roll["pbar"]
            print("p_bar: ", p_bar)
            return p_bar

    def get_k_D(self):
        # getting k_D to calculate the coefficient of induced drag
        k_D = 0
        for i in range(1, self.nodes):
            k = (i+1) * (self.a_n[i]**2/self.a_n[0]**2)
            k_D += k
        print("K_D: ", k_D)
        return k_D

    def get_k_L(self, file_name):
        # pulling in aspect ratio and lift slope
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]
        cl_alpha1 = shape["airfoil_lift_slope"]

        # getting k_L to calculate cl_alpha
        k_L = (1 - (1 + (np.pi * self.ra)/cl_alpha1) * self.a_n[0])/((1 + (np.pi * self.ra)/cl_alpha1) * self.a_n[0])
        print("K_L: ", k_L)
        return k_L

    def get_k_DL(self, file_name):
        # pulling in the washout
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        washout = shape["washout"]["distribution"]

        # getting kappa DL
        k_dl = 0
        if washout == "none":
            return k_dl
        else:
            for i in range(1, self.nodes):
                k_dl1 = (i+1) * (self.a_n[i]/self.a_n[0]) * ((self.b_n[i]/self.b_n[0]) - (self.a_n[i]/self.a_n[0]))
                k_dl += k_dl1
            k_DL = 2 * (self.b_n[0]/self.a_n[0]) * k_dl
            print("K_DL: ", k_DL)
            return k_DL

    def get_k_d_omega(self, file_name):
        # pulling in the washout
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        washout = shape["washout"]["distribution"]

        # getting kappa d omega
        k_do = 0
        if washout == "none":
            return k_do
        else:
            for i in range(1, self.nodes):
                k_dom = (i+1) * ((self.b_n[i]/self.b_n[0]) - (self.a_n[i]/self.a_n[0]))**2
                k_do += k_dom
            k_omega = (self.b_n[0]/self.a_n[0])**2 * k_do
            print("K_D_omega: ", k_omega)
            return k_omega

    def get_epsilon_o(self):
        # getting epsilon omega
        e_o = self.b_n[0]/self.a_n[0]
        print("Epsilon_omega: ", e_o)
        return e_o

    def get_es(self):
        # span efficiency factor
        es = 1/(1 + self.k_D)
        print("Span Efficiency Factor, e_s: ", es)
        return es

    def get_cl_alpha(self, file_name):
        # pulling in aspect ratio and lift slope
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]
        cl_alpha1 = shape["airfoil_lift_slope"]
        # wing lift slope
        cl_alpha = cl_alpha1/((1 + cl_alpha1/(np.pi * self.ra)) * (1 + self.k_L))
        print("CL_alpha: ", cl_alpha)
        return cl_alpha

    def get_alpha_root(self, file_name):
        # pulling in angle of attack and aileron deflection
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        condition = input_dict["condition"]
        alpha = condition["alpha_root[deg]"]

        if alpha == "CL":
            cl = condition["CL"]
            alpha_root = cl/self.cl_alpha + (self.e_o * self.omega)
            print("(alpha - alpha_L0): ", np.degrees(alpha_root))
            return alpha_root
        else:
            print("(alpha - alpha_L0): ", alpha)
            return alpha

    def get_A_n(self, file_name):
        # pulling in angle of attack and aileron deflection
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        deflection = input_dict["condition"]
        delta_a = np.radians(deflection["aileron_deflection[deg]"])
        print("Aileron[deg]: ", np.degrees(delta_a))
        # alpha = np.radians(deflection["alpha_root[deg]"])
        # alpha_l0 = 0

        A_n = np.zeros((self.nodes, 1))
        for i in range(0, self.nodes):
            A_n[i] = (self.a_n[i] * self.alpha_root) - (self.b_n[i] * self.omega) + (self.c_n[i] * delta_a) + (self.d_n[i] * self.p_bar)
        return A_n

    def get_cl_roll(self, file_name):
        # rolling-moment coefficient
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        cl_roll = -((np.pi * self.ra)/4) * self.A_n[1]
        print("Cl: ", cl_roll)
        return cl_roll

    def get_cn_yaw(self, file_name):
        # yawing-moment coefficient
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        cn = 0
        for i in range(1, self.nodes):
            cn1 = ((2 * (i+1)) - 1) * self.A_n[i-1] * self.A_n[i]
            cn += cn1
        cn_yaw = (cn * ((np.pi * self.ra)/4)) - (((np.pi * self.ra * self.p_bar)/8) * (self.A_n[0] + self.A_n[2]))
        print("Cn: ", cn_yaw)
        return cn_yaw

    def get_cl(self, file_name):
        # pulling in the angle of attack at the root of the wing (no twist currently)
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        # calculating the coefficient of lift
        cl = np.pi * self.ra * self.A_n[0]
        print("Coefficient of Lift, C_L: ", cl)
        return cl

    def get_omega(self, file_name):
        # pulling in the washout amount, big omega
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        # condition = input_dict["condition"]
        shape = input_dict["wing"]
        amount = shape["washout"]["amount[deg]"]
        distribution = shape["washout"]["distribution"]
        planform = shape["planform"]["type"]
        cl_design = shape["washout"]["CL_design"]
        b3 = shape["washout"]["B3"]
        # r_a = shape["planform"]["aspect_ratio"]
        cl_alpha = shape["airfoil_lift_slope"]
        c_root = shape["nodes_per_semispan"] - 1

        if amount == "optimum":
            if distribution == "optimum":
                omega = (cl_design/(np.pi * self.ra)) * ((4 * (1 - b3))/(cl_alpha * self.cb[c_root]) - (12 * b3))
                print("Omega[deg]: ", np.degrees(omega))
                return omega
            else:
                if planform == "elliptic":
                    omega = 0
                    print("Omega[deg]: ", np.degrees(omega))
                    return omega
                else:
                    omega = (self.k_DL * cl_design)/(2 * self.k_D_omega * self.cl_alpha)
                    print("Omega[deg]: ", np.degrees(omega))
                    return omega
        else:
            omega = np.radians(shape["washout"]["amount[deg]"])
            return omega

    def get_cD_i(self, file_name):
        # pulling in aspect ratio
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        # induced drag neglecting aileron and rolling rate
        cd_i = ((self.cl**2 * (1 + self.k_D)) - (self.k_DL * self.cl * self.cl_alpha * self.omega) + (self.k_D_omega * (self.cl_alpha * self.omega)**2))/(np.pi * self.ra)
        print("Coefficient of Induced Drag, C_Di: ", cd_i)
        return cd_i

    def get_cD_i_aileron(self, file_name):
        # pulling in aspect ratio
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        shape = input_dict["wing"]
        # r_a = shape["planform"]["aspect_ratio"]

        # induced drag with aileron deflection and rolling rate
        cdi = 0
        for i in range(0, self.nodes):
            cd0 = (i+1) * (self.A_n[i])**2
            cdi += cd0
        cd_i = (np.pi * self.ra * cdi) - (((np.pi * self.ra * self.p_bar)/2) * self.A_n[1])
        print("Induced drag with aileron, C_DI: ", cd_i)
        return cd_i

    def get_cl_hat_planform(self):
        # contribution to dimensionless lift distribution from planform
        cl_planform = []
        for i in range(self.nodes):
            cl_value = 0
            for j in range(self.nodes):
                cl_value += self.a_n[j] * np.sin((j+1) * self.theta[i])
            cl_planform.append(cl_value)

        cL_planform = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cL_planform[i] = 4 * self.alpha_root * cl_planform[i]
        return cL_planform

    def get_cl_hat_washout(self):
        # contribution to dimensionless lift distribution from washout
        cl_washout = []
        for i in range(self.nodes):
            cl_value = 0
            for j in range(self.nodes):
                cl_value += self.b_n[j] * np.sin((j + 1) * self.theta[i])
            cl_washout.append(cl_value)

        cL_washout = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cL_washout[i] = -4 * self.omega * cl_washout[i]
        return cL_washout

    def get_cl_hat_aileron(self, file_name):
        # aileron deflection
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        deflection = input_dict["condition"]
        delta_a = np.radians(deflection["aileron_deflection[deg]"])

        # contribution to dimensionless lift distribution from ailerons
        cl_aileron = []
        for i in range(self.nodes):
            cl_value = 0
            for j in range(self.nodes):
                cl_value += self.c_n[j] * np.sin((j + 1) * self.theta[i])
            cl_aileron.append(cl_value)

        cL_aileron = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cL_aileron[i] = 4 * delta_a * cl_aileron[i]
        return cL_aileron

    def get_cl_hat_roll(self):
        # contribution to dimensionless lift distribution from roll
        cl_roll = []
        for i in range(self.nodes):
            cl_value = 0
            for j in range(self.nodes):
                cl_value += self.d_n[j] * np.sin((j + 1) * self.theta[i])
            cl_roll.append(cl_value)

        cL_roll = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cL_roll[i] = 4 * self.p_bar * cl_roll[i]
        return cL_roll

    def get_cl_hat(self):
        cl_hat = self.cl_hat_planform + self.cl_hat_washout + self.cl_hat_aileron + self.cl_hat_roll
        return cl_hat

    def get_cl_tilde_planform(self):
        # contribution to the local section lift coefficient from planform
        cl_planform = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cl_planform[i] = self.cl_hat_planform[i] / self.cb[i]
        return cl_planform

    def get_cl_tilde_washout(self):
        # contribution to the local section lift coefficient from washout
        cl_washout = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cl_washout[i] = self.cl_hat_washout[i] / self.cb[i]
        return cl_washout

    def get_cl_tilde_aileron(self):
        # contribution to the local section lift coefficient from aileron
        cl_aileron = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cl_aileron[i] = self.cl_hat_aileron[i] / self.cb[i]
        return cl_aileron

    def get_cl_tilde_roll(self):
        # contribution to the local section lift coefficient from roll
        cl_roll = np.zeros((self.nodes, 1))
        for i in range(self.nodes):
            cl_roll[i] = self.cl_hat_roll[i] / self.cb[i]
        return cl_roll

    def get_cl_tilde(self):
        # total contribution to the local section lift coefficient
        cl_tilde = self.cl_tilde_planform + self.cl_tilde_washout + self.cl_tilde_aileron + self.cl_tilde_roll
        return cl_tilde

    def write_to_file(self, file_name):
        # writing to the solutions text file
        filename = "Solution.txt"
        with open(filename, "w") as f:
            # writing the c matrix out
            f.write("C Matrix\n")
            for i in range(0, self.nodes):
                for j in range(0, self.nodes):
                    f.write("{:< 16.12f}".format(self.c_mat[i, j]))
                f.write("\n")
            f.write("\n")
            # writing the c inverse matrix out
            f.write("C Inverse Matrix\n")
            for i in range(0, self.nodes):
                for j in range(0, self.nodes):
                    f.write("{:< 16.12f}".format(self.c_inv[i, j]))
                f.write("\n")
            f.write("\n")
            # writing the a_n Fourier coefficients out
            f.write("Fourier Coefficients a_n\n")
            for i in range(0, self.nodes):
                f.write("{:< 16.12f}".format(self.a_n[i, 0]))
                f.write("\n")
            f.write("\n")
            # writing the b_n Fourier coefficients out
            json_string = open(file_name).read()
            input_dict = json.loads(json_string)
            shape = input_dict["wing"]
            washout = shape["washout"]["distribution"]
            if washout == "none":
                return
            else:
                f.write("Fourier Coefficients b_n\n")
                for i in range(0, self.nodes):
                    f.write("{:< 16.12f}".format(self.b_n[i]))
                    f.write("\n")
                f.write("\n")
            # writing the c_n Fourier coefficients out
            f.write("Fourier Coefficients c_n\n")
            for i in range(0, self.nodes):
                f.write("{:< 16.12f}".format(self.c_n[i, 0]))
                f.write("\n")
            f.write("\n")
            # writing the d_n Fourier coefficients out
            f.write("Fourier Coefficients d_n\n")
            for i in range(0, self.nodes):
                f.write("{:< 16.12f}".format(self.d_n[i]))
                f.write("\n")
            f.write("\n")

            f.close()

    def get_plot_planform(self, file_name):
        # pulling in decision to plot or not to plot
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        view = input_dict["view"]
        graph = view["planform"]
        aileron = input_dict["wing"]
        zb1 = aileron["aileron"]["begin[z/b]"]
        zb2 = aileron["aileron"]["end[z/b]"]
        cfc1 = aileron["aileron"]["begin[cf/c]"]
        cfc2 = aileron["aileron"]["begin[cf/c]"]

        if graph:
            # fig = plt.figure()
            ax = plt.subplot()
            # plotting the planform
            le = self.cb * 0.25
            te = -self.cb * 0.75
            ax.plot(self.zb, le, "k")
            ax.plot(self.zb, te, "k")
            # plotting the quarter-chord line
            c4 = self.cb * 0
            ax.plot(self.zb, c4, "k")
            # plotting the values of theta
            for i in range(self.nodes):
                edges = [le[i], te[i]]
                vert = [self.zb[i], self.zb[i]]
                ax.plot(vert, edges, "b")
            # plotting the ailerons
            planform = aileron["planform"]["type"]  # reads in the type of planform
            if planform == "elliptic":
                # r_a = aileron["planform"]["aspect_ratio"]
                cb1 = (4 / (np.pi * self.ra)) * np.sqrt(1 - (2 * zb1) ** 2)  # chord distribution at z/b value
                cb2 = (4 / (np.pi * self.ra)) * np.sqrt(1 - (2 * zb2) ** 2)
                y1 = cb1 * (-0.75 + cfc1)
                y2 = cb2 * (-0.75 + cfc2)
                slope1 = (y2 - y1) / (zb2 - zb1)
                b1 = y2 - (slope1 * zb2)
                y_1 = cb1 * (-0.75)
                y_2 = cb2 * (-0.75)
                x = [zb1, zb1, zb2, zb2]
                x1 = [-zb1, -zb1, -zb2, -zb2]
                y = [y_1, y1, y2, y_2]
                ax.plot(x, y, "k")
                ax.plot(x1, y, "k")
                for i in range(self.nodes):
                    if zb1 < self.zb[i] < zb2:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")
                    if -zb2 < self.zb[i] < zb1:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (-slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")
            if planform == "tapered":
                # r_a = aileron["planform"]["aspect_ratio"]
                r_t = aileron["planform"]["taper_ratio"]
                cb1 = (2 / (self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(2 * zb1)))  # chord distribution using z/b value
                cb2 = (2 / (self.ra * (1 + r_t))) * (1 - ((1 - r_t) * np.abs(2 * zb2)))
                y1 = cb1 * (-0.75 + cfc1)
                y2 = cb2 * (-0.75 + cfc2)
                slope1 = (y2 - y1) / (zb2 - zb1)
                b1 = y2 - (slope1 * zb2)
                y_1 = cb1 * (-0.75)
                y_2 = cb2 * (-0.75)
                x = [zb1, zb1, zb2, zb2]
                x1 = [-zb1, -zb1, -zb2, -zb2]
                y = [y_1, y1, y2, y_2]
                ax.plot(x, y, "k")
                ax.plot(x1, y, "k")
                for i in range(self.nodes):
                    if zb1 <= self.zb[i] <= zb2:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")
                    if -zb2 <= self.zb[i] <= -zb1:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (-slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")

            if planform == "file":
                cb1 = np.interp(zb1, self.zb, self.cb)
                cb2 = np.interp(zb2, self.zb, self.cb)
                y1 = cb1 * (-0.75 + cfc1)
                y2 = cb2 * (-0.75 + cfc2)
                slope1 = (y2 - y1) / (zb2 - zb1)
                b1 = y2 - (slope1 * zb2)
                y_1 = cb1 * (-0.75)
                y_2 = cb2 * (-0.75)
                x = [zb1, zb1, zb2, zb2]
                x1 = [-zb1, -zb1, -zb2, -zb2]
                y = [y_1, y1, y2, y_2]
                ax.plot(x, y, "k")
                ax.plot(x1, y, "k")
                for i in range(self.nodes):
                    if zb1 <= self.zb[i] <= zb2:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")
                    if -zb2 <= self.zb[i] <= -zb1:
                        xlist = [self.zb[i], self.zb[i]]
                        ylist = [-self.cb[i] * 0.75, (-slope1 * self.zb[i]) + b1]
                        ax.plot(xlist, ylist, "r")

            # formatting the plot
            ax.set_aspect("equal", adjustable="box")
            plt.xlabel("z/b")
            plt.ylabel("c/b")
            plt.title("Planform")
            plt.show()
        else:
            return

    def get_plot_washout(self, file_name):
        # pulling in decision to plot or not to plot
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        view = input_dict["view"]
        graph2 = view["washout_distribution"]

        if graph2:
            plt.plot(self.zb, self.w, "k")
            plt.xlabel("z/b")
            plt.ylabel("omega")
            plt.title("Washout Distribution")
            plt.show()

    def get_plot_aileron(self, file_name):
        # pulling in decision to plot or not to plot
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        view = input_dict["view"]
        graph3 = view["aileron_distribution"]

        if graph3:
            plt.plot(self.zb, self.chi, "k")
            plt.xlabel("z/b")
            plt.ylabel("chi")
            plt.title("Aileron Distribution")
            plt.show()

    def get_plot_cl_hat(self, file_name):
        # pulling in decision to plot or not to plot
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        view = input_dict["view"]
        graph4 = view["CL_hat_distributions"]

        if graph4:
            ax = plt.subplot()
            ax.plot(self.zb, self.cl_hat_planform, "b", label="Planform")
            ax.plot(self.zb, self.cl_hat_washout, "g", label="Washout")
            ax.plot(self.zb, self.cl_hat_aileron, "r", label="Aileron")
            ax.plot(self.zb, self.cl_hat_roll, "m", label="Roll Rate")
            ax.plot(self.zb, self.cl_hat, "k", label="Total")

            plt.xlabel("z/b")
            plt.ylabel("CL_hat = L_tilde/(0.5*rho*V^2*b)")
            plt.title("CL_hat Distributions")
            ax.legend(loc='upper right')
            plt.show()

    def get_plot_cl_tilde(self, file_name):
        # pulling in decision to plot or not to plot
        json_string = open(file_name).read()
        input_dict = json.loads(json_string)
        view = input_dict["view"]
        graph4 = view["CL_tilde_distributions"]

        if graph4:
            ax = plt.subplot()
            ax.plot(self.zb, self.cl_tilde_planform, "b", label="Planform")
            ax.plot(self.zb, self.cl_tilde_washout, "g", label="Washout")
            ax.plot(self.zb, self.cl_tilde_aileron, "r", label="Aileron")
            ax.plot(self.zb, self.cl_tilde_roll, "m", label="Roll Rate")
            ax.plot(self.zb, self.cl_tilde, "k", label="Total")

            plt.xlabel("z/b")
            plt.ylabel("CL_tilde = L_tilde/(0.5*rho*V^2*c)")
            plt.title("CL_tilde Distributions")
            ax.legend(loc='upper right')
            plt.show()

if __name__ == "__main__":
    wing("input.json")
