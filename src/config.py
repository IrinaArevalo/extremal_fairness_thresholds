
from typing import Literal

MetricName = Literal[
    "statistical_parity_difference",  # P(ŷ=1|A<t) - P(ŷ=1|A>=t)
    "disparate_impact",              # P(ŷ=1|A<t) / P(ŷ=1|A>=t)
    "log_disparate_impact",          # log(DI) (symmetric-ish for optimization)
    "equal_opportunity_difference",  # TPR(A<t) - TPR(A>=t)
    "average_odds_difference",       # 0.5*(TPR diff + FPR diff)
    "auc_difference",                # AUC(A<t) - AUC(A>=t)
]

paths = {
     "german_uci": "data/german.data",
     "adult_uci": "data/adult.data",
     "taiwan_xls": "data/default of credit card clients.xls",
     "compas_db": "data/compas.db",
     "give_me_credit_train": "data/cs-training.csv",
 }
