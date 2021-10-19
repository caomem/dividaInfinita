import gurobipy as gp

class Modelo_cripto_SVM(object):
    def __init__(self, Sigma, y_vals, lambda_val):
        self.Sigma = Sigma
        self.y_vals = y_vals
        self.lambda_val = lambda_val
        self.n = len(self.y_vals)

        self.model = gp.Model()

        self.d = self.model.addMVar(self.n, lb=-1/(2*self.n), ub=1/(2*self.n), vtype = gp.GRB.CONTINUOUS)

        self.termo_linear_obj_fun = (self.y_vals @ self.d)
        self.termo_nao_linear_obj_fun = (self.d @ (self.Sigma/2) @ self.d)

        self.obj_fun = self.model.setObjective(
            self.lambda_val * self.termo_linear_obj_fun - self.termo_nao_linear_obj_fun, sense = gp.GRB.MAXIMIZE
        )
        
        self.c1 = self.model.addConstr(self.d.sum() == 0)

        self.y_vals_pos = self.y_vals > 0
        self.c2 = self.model.addConstr(self.d[self.y_vals_pos] >= 0)
        self.c2 = self.model.addConstr(self.d[~self.y_vals_pos] <= 0)
        del self.y_vals_pos

        self.obj_val = None

    def solve(self, time=None, heur=None, log=0):
        if(time != None):
            self.model.setParam('TimeLimit', time)
        if(heur != None):
            self.model.setParam('Heuristics', heur)
        if(log >= 0):
            try:
                self.model.Params.LogToConsole = log
            except:
                self.model.Params.LogToConsole = 1
        else:
            self.model.Params.LogToConsole = 1

        self.result = self.model.optimize()

        try:
            self.d[0].X >= 0
        except:
            print('Nenhuma solucao encontrada!')
            return 0
        
        self.obj_val = self.model.getObjective().getValue()
        self.sol_time = self.model.Runtime

        return self.d.X

    def update_lambda(self, lambda_val = 1):
        self.lambda_val = lambda_val
        try:
            self.model.remove(self.obj_fun)
        except:
            pass
        self.obj_fun = self.model.setObjective(
            self.lambda_val * self.termo_linear_obj_fun - self.termo_nao_linear_obj_fun, sense = gp.GRB.MAXIMIZE
        )