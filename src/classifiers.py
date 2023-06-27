from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.ensemble import RUSBoostClassifier, BalancedBaggingClassifier
from imblearn.pipeline import Pipeline
from self_paced_ensemble import SelfPacedEnsembleClassifier
from definitions import *

classifiers = {
    'SPE': Pipeline(steps=[('model', SelfPacedEnsembleClassifier(n_estimators=N_ESTIMATORS,
                                                                 random_state=RANDOM_STATE,
                                                                 base_estimator=base_clf))]),  # novel
    "AdaBoost": Pipeline(steps=[('model',
                                 AdaBoostClassifier(n_estimators=N_ESTIMATORS,
                                                    random_state=RANDOM_STATE,
                                                    base_estimator=base_clf))]),  # boosting
    # 'grad': Pipeline(steps=[('model',
    #                          GradientBoostingClassifier(n_estimators=N_ESTIMATORS,
    #                                                     random_state=RANDOM_STATE ))]),     # boosting
    'Bagging': Pipeline(steps=[('model',
                                BaggingClassifier(n_estimators=N_ESTIMATORS,
                                                  random_state=RANDOM_STATE,
                                                  base_estimator=base_clf))]),  # bagging
    'RUSBoost': Pipeline(steps=[('model', RUSBoostClassifier(n_estimators=N_ESTIMATORS,
                                                             random_state=RANDOM_STATE,
                                                             estimator=base_clf))]),  # boosting

    'BalancedBag': Pipeline(steps=[('model', BalancedBaggingClassifier(n_estimators=N_ESTIMATORS,
                                                                       random_state=RANDOM_STATE,
                                                                       estimator=base_clf))]),  # bagging

    'SMOTE_Bag': Pipeline(steps=[('over', SMOTE()),
                                 ('model', BaggingClassifier(n_estimators=N_ESTIMATORS,
                                                             random_state=RANDOM_STATE,
                                                             base_estimator=base_clf))]),  # bagging

}
