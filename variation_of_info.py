import numpy as np
from munkres import Munkres, print_matrix
from collections import Counter
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from ast import literal_eval
import os


def compute_cost(zt, zt_real):
    cost_mat = []  # np.zeros((len(np.unique(zt_real)), len(np.unique(zt))));
    K_use = max(len(np.unique(zt_real)), len(np.unique(zt)))
    for ii in range(K_use):  ## real
        cost_mat.append([])
        for jj in range(K_use):
            cost_mat[ii].append((np.abs((zt_real == ii) * 1 - (zt == jj) * 1)).sum())
    # print_matrix(cost_mat);

    m = Munkres()
    indexes = m.compute(cost_mat)

    total = 0
    for row, column in indexes:
        value = cost_mat[row][column]
        total += value
        # print(f'({row}, {column}) -> {value}')
    # print(f'total cost: {total}')
    return total, indexes


def euclidean_distance(A, B):
    assert A.shape == B.shape, "two matrices should have the same shape"
    return np.sqrt(np.sum((A - B) ** 2))


def difference(A, B):
    assert len(A) == len(B), "two hidden states set should have the same length"
    miss_sum = 0
    tot_num = 0
    for i in range(len(A)):
        assert len(A[i]) == len(B[i])
        miss_sum += np.sum(np.array(A[i]) != np.array(B[i]))
        tot_num += len(A[i])
    return miss_sum, tot_num


def kl_divergence(P, Q):
    assert P.shape == Q.shape, "two matrices should have the same shape"
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))


def viterbi(observations, num_states, transition_prob, emission_prob):
    V = np.zeros((num_states, len(observations)))
    path = {}
    # first column of V is the transition prob from state 0
    V[:, 0] = transition_prob[0, :]

    for t in range(1, len(observations)):
        for s in range(1, num_states):
            prob = V[:, t - 1] * transition_prob[:, s] * emission_prob[s - 1, observations[t]]
            V[s, t] = np.max(prob)
            path[s, t] = np.argmax(prob)

    optimal_path = []
    last_state = np.argmax(V[:, -1])
    optimal_path.append(last_state)

    for t in range(len(observations) - 1, 1, -1):
        last_state = path[last_state, t]
        optimal_path.insert(0, last_state)

    optimal_path.insert(0, 0)

    return optimal_path


def set_print_options():
    np.set_printoptions(suppress=True, precision=4)
    np.set_printoptions(linewidth=180)
    np.set_printoptions(formatter={'int': '{:5d}'.format})

def flatten(lists):
    return [element for l in lists for element in l]


def calculate_entropy(cluster):
    """Calculate the entropy of a clustering."""
    total_points = len(cluster)
    if total_points == 0:
        return 0
    label_counts = Counter(cluster)
    probabilities = [count / total_points for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def calculate_mutual_information(U, V):
    """Calculate the mutual information between two clusterings."""
    total_points = len(U)
    mutual_info = 0
    U_labels, V_labels = set(U), set(V)
    for u in U_labels:
        for v in V_labels:
            intersection_size = sum(1 for i in range(total_points) if U[i] == u and V[i] == v)
            if intersection_size == 0:
                continue
            p_u = sum(1 for x in U if x == u) / total_points
            p_v = sum(1 for x in V if x == v) / total_points
            p_uv = intersection_size / total_points
            mutual_info += p_uv * np.log2(p_uv / (p_u * p_v))
    return mutual_info

def calculate_variation_of_information(U, V):
    """Calculate the variation of information between two clusterings."""
    entropy_U = calculate_entropy(U)
    entropy_V = calculate_entropy(V)
    mutual_information = calculate_mutual_information(U, V)
    variation_of_information = entropy_U + entropy_V - 2 * mutual_information
    return variation_of_information, variation_of_information / (entropy_U + entropy_V)

def calculate_v_measure(true_labels, predicted_labels):
    homo_score = homogeneity_score(true_labels, predicted_labels)
    comp_score = completeness_score(true_labels, predicted_labels)
    v_score = v_measure_score(true_labels, predicted_labels)
    return homo_score, comp_score, v_score

def list_subdirectories(directory_path):
    return [name for name in os.listdir(directory_path)
            if os.path.isdir(os.path.join(directory_path, name))]

path_to_tl = r"/Users/milamarcheva/Downloads/all_tags.txt"
file1 = open(path_to_tl, 'r')
tl= file1.readlines()
file1.close()
tl = [l.strip().split() for l in tl]
tl = [i for j in tl for i in j]

# print(tl[:3])
# print(type(tl[0]))


path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/orderedNormalTest.out/stage2.current.test.pred.0"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/shuffledNormalTest.out/stage2.current.test.pred.0"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/orderedMorphTest.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/shuffledMorphTest.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/BrownShuffledStagesNoGradualUnlocking.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/BrownShuffledNoStagesWithGradualUnlocking.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/BrownOrderedStagesNoGradualUnlocking.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/BrownOrderedNoStagesWithGradualUnlocking.out/test.current.test.pred.test_all"
path_to_pl = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/23042024/MorphBrownStages_dilute2_order5555_grouping5_PREPAnchorIndex0_n10_v10_0_1_1_1_1_4_2.out/test.current.test.pred.test_all"
path_dir_anchor1 = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/06012025/"
path_dir_anchor1 = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/line4/"
path_dir_anchor1 = r"/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/line5/"
path_dir_anchor1 = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/line7/"
path_dir_anchor1 = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/out2025/line8/"

file1 = open(path_to_pl, 'r')
pl= file1.readlines()
file1.close()
pl = [l.strip().split() for l in pl]
pl = [i for j in pl for i in j]

# print(pl[:3])

# print(calculate_v_measure(tl,pl))
# print(calculate_variation_of_information(tl,pl))

def read_file(path_to_pl):
    file1 = open(path_to_pl, 'r')
    pl= file1.readlines()
    file1.close()
    pl = [l.strip().split() for l in pl]
    pl = [i for j in pl for i in j]

    return pl

def get_anchor1_stats():
    sub_to_vi = {}
    subdirs = list_subdirectories(path_dir_anchor1)
    for sub in subdirs:
        path_to_pl = path_dir_anchor1+sub+"/test.current.test.pred.test_all"
        pl = read_file(path_to_pl)
        sub_to_vi[sub] = calculate_variation_of_information(tl,pl)[0]

    all_vi_vals = list(sub_to_vi.values())
    print(all_vi_vals)
    print('mean: ', np.average(all_vi_vals), ' std: ', np.std(all_vi_vals))


get_anchor1_stats()

