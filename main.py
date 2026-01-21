import time
import instance_parsing as ps
import utils as ut
import greedy as gs
import problem_gurobi as gb
import iterative_heuristics as it_h
import local_search as ls

def run_tests(file_name):
    # --- CHARGEMENT ET PRÉPARATION ---
    data = ps.parse_file(file_name)
    Lcoord, items, capacity, vMin, vMax, K = data
    Mdist = ut.distance_matrix(Lcoord)
    
    # Paramètres pour Gurobi et Greedy
    params = gs.init(file_name)
    P_gr, V_gr, nb_v, nb_obj = params[0], params[1], params[2], params[3]
    v_start = params[7]

    # Tour de base pour les heuristiques (1-based pour EA)
    raw_tour = ut.LK_solve(file_name)
    tour_1_based = [t + 1 for t in raw_tour]
    if tour_1_based[-1] != tour_1_based[0]:
        tour_1_based.append(tour_1_based[0])

    # ==========================================
    # SECTION 1 : KCTSP (5 Algorithmes)
    # ==========================================
    scores_k = []
    times_k = []

    # 1. Gurobi MTZ
    start = time.time()
    res_mtz = gb.solve_problem(P_gr, V_gr, nb_v, nb_obj, Mdist, capacity, K, v_start, 
                               verbose=False, gurobi_out=False, time_limit=20, use_mtz=True)
    times_k.append(time.time() - start)
    scores_k.append(res_mtz[1] if res_mtz else 0)

    # 2. Gurobi GG
    start = time.time()
    res_gg = gb.solve_problem(P_gr, V_gr, nb_v, nb_obj, Mdist, capacity, K, v_start, 
                              verbose=False, gurobi_out=False, time_limit=20, use_mtz=False)
    times_k.append(time.time() - start)
    scores_k.append(res_gg[1] if res_gg else 0)

    # 3. Greedy (KCTSP)
    start = time.time()
    _, score_gr_k, _, _, _, _ = gs.solve_greedy(file_name, TTP=False)
    times_k.append(time.time() - start)
    scores_k.append(score_gr_k)

    # 4. EA (KCTSP)
    start = time.time()
    _, score_ea_k, _ = it_h.EA_KCTSP_incremental(items, tour_1_based, Mdist, K, capacity, iterations=1000)
    times_k.append(time.time() - start)
    scores_k.append(score_ea_k)

    # 5. Local Search (KCTSP)
    start = time.time()
    _, score_ls_k, _, _, _ = ls.solve_ls(file_name, TTP=False)
    times_k.append(time.time() - start)
    scores_k.append(score_ls_k)

    # ==========================================
    # SECTION 2 : TTP (2 Algorithmes)
    # ==========================================
    scores_t = []
    times_t = []

    # 1. Greedy (TTP)
    start = time.time()
    _, score_gr_t, _, _, _, _ = gs.solve_greedy(file_name, TTP=True)
    times_t.append(time.time() - start)
    scores_t.append(score_gr_t)

    # 2. EA (TTP)
    start = time.time()
    _, score_ea_t, _ = it_h.EA_TTP_incremental(items, tour_1_based, Mdist, K, vMin, vMax, capacity, iterations=1000)
    times_t.append(time.time() - start)
    scores_t.append(score_ea_t)

    return [[scores_k, times_k], [scores_t, times_t]]

if __name__ == "__main__":
    instance = "inst20.tsp"
    try:
        results = run_tests(instance)
        
        k_res, t_res = results
        
        print(f"=== COMPARAISON SUR {instance} ===")
        print("\n--- KCTSP (MTZ, GG, Greedy, EA, LS) ---")
        print(f"Scores : {[round(s, 2) for s in k_res[0]]}")
        print(f"Temps  : {[round(t, 4) for t in k_res[1]]} s")
        
        print("\n--- TTP (Greedy, EA) ---")
        print(f"Scores : {[round(s, 2) for s in t_res[0]]}")
        print(f"Temps  : {[round(t, 4) for t in t_res[1]]} s")
       
    except FileNotFoundError:
        print(f"Erreur : Vérifiez la présence de {instance} dans le dossier ./instances/")
