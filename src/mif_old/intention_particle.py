import numpy as np

class IntentionParticle:
    def __init__(self, intention_sampler, particle_num_per_intent=200, num_tpp=12):
        """
        input:
            - intention_mean
            - intention_num
                the number of intentions.
            - particle_num_per_intent
                the number of particles set for each intention.
            - num_tpp
                number of trajectory points for prediction.
        """
        self.intention_sampler = intention_sampler
        self.intention_num = self.intention_sampler.intent_num
        self.intention_coordinates = self.intention_sampler.intent_bottom_left_coordinates + self.intention_sampler.intent_wid
        self.intention = np.arange(self.intention_num).reshape(-1, 1).dot(\
            np.ones(particle_num_per_intent).reshape(1,-1)).reshape(-1).astype(int) # [0,0,...,0,1,1,...,1,2,2,...,2] (particle_num,)
        self.num_tpp = num_tpp
        self.intention_mask = self.create_intention_mask()
        self.particle_num = self.intention_num * particle_num_per_intent
        self.reset()
    
    def create_intention_mask(self):
        intention_mask = []
        for intention_index in range(self.intention_num):
            intention_mask.append(self.intention == intention_index)
        return np.vstack(intention_mask) #(intention_num, particle_num)

    def reset(self):
        self.weight = np.ones(self.particle_num) * (1./self.particle_num) # (particle_num,)
    
    def resample(self):
        ### soft weight ###
        # weight_balanced = np.log(self.weight+params.WEIGHT_EPS) - np.log(min(self.weight+params.WEIGHT_EPS)) + 1. # (no zero in it is convenient for sampling)
        # weight_balanced /= sum(weight_balanced)
        ### original weight ###
        weight_balanced = self.weight
        ######
        resampled_indices = np.random.choice(self.particle_num, size=self.particle_num, p=weight_balanced)
        self.intention = self.intention[resampled_indices] # inherited
        self.intention_mask = self.create_intention_mask()
        self.reset()
    
    def mutate(self, mutation_prob=0.01):
        # if mutation_prob=0.01, for example, current intention is 1, the chance of the intention being mutated is 0.01.
        # 0.005 percent to intention 2, 0.005 percent to intention 3.
        # In the algorithm, it is 0.985 to keep as it is, 0.005 for each possible intention.
        mutation_mask_prob = 1.-mutation_prob*self.intention_num/(self.intention_num-1.) # like 0.985 for mutation_prob=0.01
        mutation_mask = np.random.uniform(size=self.particle_num) > mutation_mask_prob
        self.intention = self.intention * (1-mutation_mask) \
            + np.random.randint(0,self.intention_num,self.particle_num) * mutation_mask
        self.intention = self.intention.astype(int)
        self.intention_mask = self.create_intention_mask()
    
    def update_weight(self, x_pred_true, tau=0.1):
        oe = self.x_pred - x_pred_true[np.newaxis]# offset error (particle_num, num_tpp, 2)
        aoe = np.mean(np.linalg.norm(oe, axis=2), axis=1) # (particle_num,)
        self.weight *= np.exp(-tau*aoe)
        self.weight /= self.weight.sum()
    
    def predict(self, x_obs, pred_func):
        self.goals = self.intention2goal()
        self.x_pred, _ = pred_func(x_obs, self.goals, self.intention, self.intention_coordinates, intention_num=self.intention_num, num_tpp=self.num_tpp)
        return self.x_pred
    
    def predict_till_end(self, x_obs, long_pred_func):
        self.goals = self.intention2goal()
        _, infos = long_pred_func(x_obs, self.goals, self.intention, self.intention_coordinates,  intention_num=self.intention_num, num_tpp=self.num_tpp)
        return infos
        

    def particle_weight_intention_prob_dist(self):
        ### soft weight ###
        # weight_balanced = np.log(self.weight+params.WEIGHT_EPS) - np.log(min(self.weight+params.WEIGHT_EPS)) + 1. # (no zero in it is convenient for sampling)
        # weight_balanced /= sum(weight_balanced)
        ### original weight ###
        weight_balanced = self.weight
        particle_weight = [] # list of intention_num (3) lists
        particle_weight_with_zero = self.weight * self.intention_mask# (particle_num,) * (intention_num, particle_num) -> (intention_num, particle_num)
        for intention_index in range(self.intention_num):
            particle_weight_intention = particle_weight_with_zero[intention_index, particle_weight_with_zero[intention_index, :].nonzero()]
            particle_weight.append(np.squeeze(particle_weight_intention, axis=0))
        return particle_weight, (weight_balanced*self.intention_mask).sum(axis=1)
    
    def intention2goal(self):
        return self.intention_sampler.idx2intent_sampling(self.intention)
