
class params_by_name:
    def __init__(self, values, names, count_par = 10,
                    **kwargs
                    ):
        self.weights = []
        i_weights = 1
        for key, value in zip(names, values):
            if 'w' in key:
                if key.index('w') == 0:
                    self.weights.append(value)
                    i_weights+=1
            if key == "undoing_threshold":
               self.undoing_threshold = value
            elif key == "undo_inverse_temparature":
               self.undo_inverse_temparature = value
            elif key == "feature_dropping_rate":
               self.feature_dropping_rate = value
            elif key == "stopping_probability":
               self.stopping_probability = value
            elif key == "pruning_threshold":
               self.pruning_threshold = value
            elif key == "lapse_rate":
               self.lapse_rate = value
            elif key == "ucb_confidence":
               self.ucb_confidence = value

            elif key == "undoing_threshold_wu":
               self.undoing_threshold_wu = value
            elif key == "undo_inverse_temparature_wu":
                self.undo_inverse_temparature_wu = value
            elif key == "feature_dropping_rate_wu":
                self.feature_dropping_rate_wu = value
            elif key == "stopping_probability_wu":
                self.stopping_probability_wu = value
            elif key == "pruning_threshold_wu":
                self.pruning_threshold_wu = value
            elif key == "lapse_rate_wu":
                self.lapse_rate_wu = value
            elif key == "ucb_confidence_wu":
                self.ucb_confidence_wu = value
        self.count_par = count_par
        self.i_th = None
        self.visits = None