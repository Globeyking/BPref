import gurobipy as gp
import utils as ut
import instance_parsing as ps

def solve_problem(
    P, V, nb_villes, nb_objets, Mdist,
    capacity, K, ville_depart,
    verbose, gurobi_out, time_limit,
    use_mtz=True
):
    m = gp.Model()
    m.Params.TimeLimit = time_limit
    if not gurobi_out:
        m.Params.OutputFlag = 0

    x, y, W = [], [], []
    load_arc = {}

    # ----------------- VARIABLES -------------------

    # MTZ variables
    u = []

    # GG variables
    g = {}

    for i in range(nb_villes):
        if use_mtz:
            # MTZ order variable
            u.append(m.addVar(lb=1, ub=nb_villes, vtype='I', name=f"u_{i}"))

        x_i = []
        for j in range(nb_villes):
            if i != j:
                x_i.append(m.addVar(vtype='B', name=f"x_{i}{j}"))
                load_arc[i, j] = m.addVar(lb=0, ub=capacity, name=f"L_{i}{j}")

                if not use_mtz:
                    # GG flow variable
                    g[i, j] = m.addVar(lb=0, ub=nb_villes - 1, name=f"g_{i}{j}")
            else:
                x_i.append(0)
        x.append(x_i)

        y.append([m.addVar(vtype='B', name=f"y_{i}_{j}") for j in range(nb_objets[i])])
        W.append(m.addVar(lb=0, ub=capacity, name=f"W_{i}"))

    # ----------------- CONTRAINTES -------------------

    # Capacity
    m.addConstr(
        gp.quicksum(V[i][j] * y[i][j] for i in range(nb_villes) for j in range(nb_objets[i]))
        <= capacity
    )

    # Initial load
    m.addConstr(
        W[ville_depart]
        == gp.quicksum(V[ville_depart][j] * y[ville_depart][j]
                       for j in range(nb_objets[ville_depart]))
    )

    # ----------------- WEIGHT & LINEARIZATION -------------------

    for i in range(nb_villes):
        for j in range(nb_villes):
            if i != j:
                # Linearization
                m.addConstr(load_arc[i, j] <= capacity * x[i][j])
                m.addConstr(load_arc[i, j] >= W[i] - capacity * (1 - x[i][j]))

                # Weight propagation
                if j != ville_depart:
                    items_j = gp.quicksum(V[j][k] * y[j][k] for k in range(nb_objets[j]))
                    m.addConstr(
                        W[j] >= W[i] + items_j - capacity * (1 - x[i][j])
                    )

    # ----------------- SUBTOUR ELIMINATION -------------------

    if use_mtz:
        # ===== MTZ =====
        for i in range(nb_villes):
            for j in range(nb_villes):
                if i != j and i != ville_depart and j != ville_depart:
                    m.addConstr(
                        u[j] >= u[i] + 1 - nb_villes * (1 - x[i][j])
                    )

    else:
        # ===== GAVISH-GRAVES =====

        # Flow out of depot
        m.addConstr(
            gp.quicksum(g[ville_depart, j]
                        for j in range(nb_villes) if j != ville_depart)
            == nb_villes - 1
        )

        for i in range(nb_villes):
            if i != ville_depart:
                m.addConstr(
                    gp.quicksum(g[j, i] for j in range(nb_villes) if j != i)
                    - gp.quicksum(g[i, j] for j in range(nb_villes) if j != i)
                    == 1
                )

            for j in range(nb_villes):
                if i != j:
                    m.addConstr(g[i, j] <= (nb_villes - 1) * x[i][j])

    # ----------------- TSP DEGREE CONSTRAINTS -------------------

    for j in range(nb_villes):
        m.addConstr(
            gp.quicksum(x[i][j] for i in range(nb_villes) if i != j) == 1
        )

    for i in range(nb_villes):
        m.addConstr(
            gp.quicksum(x[i][j] for j in range(nb_villes) if i != j) == 1
        )

    # ----------------- OBJECTIVE -------------------

    obj_profit = gp.quicksum(
        P[i][j] * y[i][j]
        for i in range(nb_villes)
        for j in range(nb_objets[i])
    )

    obj_renting = K * gp.quicksum(
        Mdist[i][j] * load_arc[i, j]
        for i in range(nb_villes)
        for j in range(nb_villes)
        if i != j
    )

    m.setObjective(obj_profit - obj_renting, gp.GRB.MAXIMIZE)
    m.optimize()

    # ----------------- SOLUTION -------------------

    best = (m.Status == gp.GRB.OPTIMAL)
    if m.SolCount == 0:
        return None

    path = [ville_depart]
    cur = ville_depart
    while True:
        nxt = next(
            (j for j in range(nb_villes)
             if cur != j and not isinstance(x[cur][j], int) and x[cur][j].X > 0.5),
            None
        )
        if nxt is None or nxt == ville_depart:
            break
        cur = nxt
        path.append(cur)

    picked_items = [
        {'city': i + 1, 'id': j + 1, 'p': P[i][j], 'w': V[i][j]}
        for i in range(nb_villes)
        for j in range(nb_objets[i])
        if y[i][j].X > 0.5
    ]

    return (
        [p + 1 for p in path],
        m.objVal,
        obj_profit.getValue(),
        obj_renting.getValue(),
        picked_items,
        best
    )

