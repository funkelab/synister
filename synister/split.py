import pylp

def find_optimal_split(synapse_ids,
                       superset_by_synapse_id,
                       nt_by_synapse_id,
                       neurotransmitters,
                       supersets,
                       train_fraction=0.8,
                       ensure_non_empty=True):
    """Find optimal synapse split per neurotransmitter 
       and synapse superset (e.g. hemi lineage/neuron/brain region)

    Args:

        synapse_ids (List):

            Synapse ids to consider
        
        superset_by_synapse_id (dict):

            Mapping from each synapse_id in synapse_ids to its
            associated superset.

        nt_by_synapse_id (dict):

            Mapping from each synapse_id in synapse_ids to
            its associated nt.

        neurotransmitters (List of tuples):

            List of neurotransmitters to consider.

        supersets (List of objects):

            List of supersets to consider.

        train_fraction (float):

            Fraction of synapses to assign to training
    """

    # Constuct combined dict:
    synapses_by_superset_and_nt = {
        (ss, nt): []
        for ss in supersets
        for nt in neurotransmitters
    }

    synapse_ids_by_nt = {nt: [] for nt in neurotransmitters}
    for synapse_id in synapse_ids:
        ss = superset_by_synapse_id[synapse_id]
        nt = nt_by_synapse_id[synapse_id]
        synapses_by_superset_and_nt[(ss, nt)].append(synapse_id)
        synapse_ids_by_nt[nt].append(synapse_id)

    # find optimal 80/20 split

    num_variables = 0
    train_indicators = {}
    target = {}
    sum_synapses = {}
    slack_u = {}
    slack_l = {}
    constraints = pylp.LinearConstraints()

    # for each NT:
    for nt in neurotransmitters:

        # compute target value: target_NT
        target[nt] = int(train_fraction*len(synapse_ids_by_nt[nt]))

        # let s_NT be sum of synapses in training for NT
        sum_synapses[nt] = num_variables
        num_variables += 1

        # measure distance to target_NT: d_NT = s_NT - target_NT
        sum_constraint = pylp.LinearConstraint()

        # for each HL:
        for ss in supersets:

            # add indicator for using (HL, NT) in train: i_HL_NT
            i = num_variables
            num_variables += 1
            train_indicators[(ss, nt)] = i

            # i_HL_NT * #_of_synapses...
            sum_constraint.set_coefficient(
                i,
                len(synapses_by_superset_and_nt[(ss, nt)]))

        # ... - s_NT = 0
        sum_constraint.set_coefficient(sum_synapses[nt], -1)
        sum_constraint.set_relation(pylp.Relation.Equal)
        sum_constraint.set_value(0)
        constraints.add(sum_constraint)

        # add two slack variables for s_NT:
        slack_u[nt] = num_variables
        num_variables += 1
        slack_l[nt] = num_variables
        num_variables += 1

        # su_NT ≥  d_NT = s_NT - target_NT              su_NT ≥ 0
        # target_NT ≥  d_NT - su_NT = s_NT - su_NT      su_NT ≥ 0

        # sl_NT ≥ -d_NT = target_NT - s_NT              sl_NT ≥ 0
        # -target_NT ≥ -d_NT - sl_NT = -s_NT - sl_NT    sl_NT ≥ 0

        slack_constraint_u = pylp.LinearConstraint()
        slack_constraint_l = pylp.LinearConstraint()
        slack_constraint_u_0 = pylp.LinearConstraint()
        slack_constraint_l_0 = pylp.LinearConstraint()

        slack_constraint_u.set_coefficient(sum_synapses[nt], 1)
        slack_constraint_u.set_coefficient(slack_u[nt], -1)
        slack_constraint_u.set_relation(pylp.Relation.LessEqual)
        slack_constraint_u.set_value(target[nt])

        slack_constraint_u_0.set_coefficient(slack_u[nt], 1)
        slack_constraint_u_0.set_relation(pylp.Relation.GreaterEqual)
        slack_constraint_u_0.set_value(0)

        slack_constraint_l.set_coefficient(sum_synapses[nt], -1)
        slack_constraint_l.set_coefficient(slack_l[nt], -1)
        slack_constraint_l.set_relation(pylp.Relation.LessEqual)
        slack_constraint_l.set_value(-target[nt])

        slack_constraint_l_0.set_coefficient(slack_l[nt], 1)
        slack_constraint_l_0.set_relation(pylp.Relation.GreaterEqual)
        slack_constraint_l_0.set_value(0)

        constraints.add(slack_constraint_u)
        constraints.add(slack_constraint_l)
        constraints.add(slack_constraint_u_0)
        constraints.add(slack_constraint_l_0)

    # ensure that either all or none of the NTs per hemi-lineages are used for
    # training
    for ss in supersets:

        prev_nt = None
        for nt in neurotransmitters:

            if prev_nt is not None:

                joint_constraint = pylp.LinearConstraint()

                joint_constraint.set_coefficient(
                    train_indicators[(ss, prev_nt)], 1)
                joint_constraint.set_coefficient(
                    train_indicators[(ss, nt)], -1)
                joint_constraint.set_relation(pylp.Relation.Equal)
                joint_constraint.set_value(0)

                constraints.add(joint_constraint)

            prev_nt = nt

    # Ensure that at least one superset is in test for each nt
    if ensure_non_empty:
        for nt in neurotransmitters:
            non_zero_constraint = pylp.LinearConstraint()
            non_one_constraint = pylp.LinearConstraint()

            # Compute number of supersets in nt:
            non_zero_ss = 0
            for ss in supersets:
                if synapses_by_superset_and_nt[(ss, nt)]:
                    non_zero_ss += 1

            # If there are more than one superset
            # require that the number of supersets
            # in test and train is at least 1
            if non_zero_ss > 1:
                for ss in supersets:
                    if synapses_by_superset_and_nt[(ss, nt)]:
                        non_zero_constraint.set_coefficient(
                                train_indicators[(ss, nt)], 1)
                        non_one_constraint.set_coefficient(
                                train_indicators[(ss, nt)], 1)

                non_zero_constraint.set_relation(pylp.Relation.GreaterEqual)
                non_zero_constraint.set_value(1)

                non_one_constraint.set_relation(pylp.Relation.LessEqual)
                non_one_constraint.set_value(non_zero_ss - 1)

                constraints.add(non_zero_constraint)
                constraints.add(non_one_constraint)

    # add sl_NT + su_NT to objective
    objective = pylp.LinearObjective(num_variables)
    for nt in neurotransmitters:
        objective.set_coefficient(slack_u[nt], 1./target[nt])
        objective.set_coefficient(slack_l[nt], 1./target[nt])

    variable_types = pylp.VariableTypeMap()
    for nt in neurotransmitters:
        variable_types[slack_u[nt]] = pylp.VariableType.Integer
        variable_types[slack_l[nt]] = pylp.VariableType.Integer
        variable_types[sum_synapses[nt]] = pylp.VariableType.Integer

    solver = pylp.create_linear_solver(pylp.Preference.Gurobi)
    solver.initialize(num_variables, pylp.VariableType.Binary, variable_types)

    solver.set_objective(objective)
    solver.set_constraints(constraints)
    solution, msg = solver.solve()

    print(msg)

    train_synapses_by_ss = {}
    test_synapses_by_ss = {}

    for nt in neurotransmitters:

        print(
            nt,
            float(solution[sum_synapses[nt]])/len(synapse_ids_by_nt[nt]), "% ",
            solution[sum_synapses[nt]], '/', len(synapse_ids_by_nt[nt]),
            '(', target[nt], ')')

        for ss in supersets:
            if len(synapses_by_superset_and_nt[(ss, nt)]) > 0:
                if solution[train_indicators[(ss, nt)]] > 0.5:
                    
                    if ss in list(train_synapses_by_ss):
                        pass
                    else:
                        train_synapses_by_ss[ss] = []

                    train_synapses_by_ss[ss].extend(synapses_by_superset_and_nt[(ss,nt)])
                    print('+', ss)
                else:
                    if ss in list(test_synapses_by_ss):
                        pass
                    else:
                        test_synapses_by_ss[ss] = []

                    test_synapses_by_ss[ss].extend(synapses_by_superset_and_nt[(ss,nt)])
                    print('-', ss)

    return train_synapses_by_ss, test_synapses_by_ss
