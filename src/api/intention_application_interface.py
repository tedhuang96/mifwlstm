import numpy as np

class IntentionApplicationInterface:
    
    def __init__(self, application):
        """
        A parent class for general intention application interface. 

        Initialize with name of the application.

        Inputs:
            - application: Name of the application. e.g. 'pedestrian2D'.

        Updated:
            - self.application
        
        Outputs:
            - None
        """
        self.application = application
        return
    
    def initialize_x(self):
        """
        Initialize the state estimate. The default output is None.

        Inputs:
            - None

        Updated:
            - None
        
        Outputs:
            - x_est: list of length :math:`num_particles` or None. The state estimates of particles. 
        """
        x_est = None
        return x_est
    
    def propagate_x(self, x_est, intention, intention_mask, x_obs=None):
        """
        Propagate state estimate forward with intention hypotheses. The default output is None.

        Inputs:
            - x_est: list of length :math:`num_particles` or None. The state estimates of particles. 
            - intention: numpy. :math:`(num_particles,)` Intention hypotheses for all particles. 
            e.g. for num_intentions=3, num_particles_per_intention=5, intention = 
            array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).
            - intention_mask: numpy. :math:`(num_intentions, num_particles)` Mask on intention 
            hypotheses of all particles.
            e.g. for intention = array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
            intention_mask = \n
            array([[ True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],\n
                   [False, False, False, False, False,  True,  True,  True,  True,  True, False, False, False, False, False],\n
                   [False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True]])
            - x_obs: None or Observation in the past.
        Updated:
            - None
        
        Outputs:
            - x_est
        """
        x_est = None
        return x_est
    
    def compare_observation(self, x_est, x_obs):
        """
        Compute the difference between the state estimate against observation. 
        The default output is zero array.

        Inputs:
            - x_est: list of length :math:`num_particles` or None. The state estimates of particles. 
            - x_obs: Observation.

        Updated:
            - None
        
        Outputs:
            - gap: numpy. :math:`(num_particles,)`. The difference between x_est and x_obs.
        """
        if x_est is None:
            raise RuntimeError('x_est is None during comparison against x_obs.')
        elif isinstance(x_est, list):
            num_particles = len(x_est)
            gap = np.zeros((num_particles))
            return gap
        else:
            raise RuntimeError('x_est is not None nor list. '\
                              +'You should use compare_observation() in the child class.')

    def resample_x(self, x_est, resampled_indices):
        """
        Use the resampled indices to re-organize the state estimates.

        Inputs:
            - x_est: list of length :math:`num_particles` or None. The state estimates of particles. 
            - resampled_indices: numpy. :math:`(num_particles,)` The resampled integer indices of 
            particles.

        Updated:
            - None
        
        Outputs:
            - resampled_x_est: list of length :math:`num_particles` or None. Organized x_est 
            according to resampled indices.
        """
        if x_est is None:
            return None
        elif isinstance(x_est, list):
            resampled_x_est = [x_est[i] for i in resampled_indices]
            return resampled_x_est
        else:
            raise RuntimeError('x_est is not None nor list. '\
                              +'You should use resample_x() in the child class.')