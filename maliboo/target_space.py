from numbers import Number
import numpy as np
import pandas as pd
from .util import ensure_rng
import inspect
from scipy.stats import qmc


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)

    Parameters
    ----------
    target_func: function, optional (default=None)
        Target function to be maximized.

    pbounds: dict, optional (default = None)
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    random_state: int, RandomState, or None, optional (default=None)
        Optionally specify a seed for a random number generator

    dataset: str, file handle, or pandas.DataFrame, optional (default=None)
        The dataset, if any, which constitutes the optimization domain (X) and possibly
        the list of target values (Y)

    target_column: str, optional (default=None)
        Name of the column that will act as the target value of the optimization.
        Only works if dataset is passed.
    
    barrier_func: dict, optional (default = None)
        Dictionary of barrier constraint functions in case of Expected Barrier or Expected Improvement methods

    debug: bool, optional (default=False)
        Whether or not to print detailed debugging information
    """
    def __init__(self, target_func=None, pbounds=None, random_state=None,
                 dataset=None, target_column=None, barrier_func = None, debug=False):
        if pbounds is None and barrier_func is None:
            raise ValueError("pbounds must be specified")
    

        self._debug = debug

        # Create an array with parameters bounds
        if pbounds is not None: 
            # Get the name of the parameters, aka the optimization variables/columns
            self._keys = sorted(pbounds)
            self._bounds = np.array(
                [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],  
                dtype=float
            )
        # pbounds.items() returns a list of tuples with key as the first element and the value as the second.
        # sorted will sort the keys in ascending order
            
        if barrier_func is not None:
            # Calculate the parameters of the first barrier function (which are the same for all barrier functions)
            self._barrier_functions = list(barrier_func.values())
            parameters = inspect.signature(self._barrier_functions[0]).parameters
            # parameters will output an ordered dictionary like this:
            # OrderedDict([('a', <Parameter "a = 5">), ('b', <Parameter "b=10">),...
            # we are only interested in the keys here:
            self._keys =  list(parameters.keys())
            # self._bounds = None

        self.x_grid = None
            
        if barrier_func is not None:
            self.barrier_func = barrier_func
        else:
            self.barrier_func = None

        
        if self._debug: print("Initializing TargetSpace with bounds:", pbounds)

        # Initialize other members
        self.seed = random_state

        self.random_state = ensure_rng(random_state)
        self.target_func = target_func
        self.initialize_dataset(dataset, target_column)

        # preallocated memory for X and Y points
        self._params = pd.DataFrame()
        self._target = np.empty(shape=(0))
        if self.barrier_func is not None:

            '''
            barrier_targets will contain the evaluation of the barrier functions, in particular
            a matrix of shape (n_evaluations, n_constraints)
            '''

            self._barrier_targets = np.empty(shape = (0,len(self.barrier_func)))
        # Other information to be recorded
        self._target_dict_info = pd.DataFrame()
        self._optimization_info = pd.DataFrame()

        if self._debug: print("TargetSpace initialization completed")

        self.in_constraint = list()
        self.n_warmup = 30

        self.n_iter = 5

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)
    
    def set_grid_dimension(self,n):
        self.n_warmup = n
    
    def set_n_iter(self,iters):
        self.n_iter = iters

    
    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)
            
    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    @property
    def dataset(self):
        return self._dataset

    @property
    def target_column(self):
        return self._target_column

    @property
    def indexes(self):
        return self._params.index
    
    @property
    def barriers(self):
        if self.barrier_func is not None:
            return self.barrier_func
        else:
            raise ValueError("barrier_func not defined")
        
    @property
    def target_barriers(self):
        if self.barrier_func is not None:
            return self._barrier_targets
        else:
            raise ValueError("target_barriers not defined")
        
    # number of constraints
    @property
    def n_constraints(self):
        if self.barrier_func is not None:
            return len(self._barrier_functions)
        else:
            raise ValueError("target_barriers not defined")
        
    def create_grid(self, n_warmup):
        
        l_bounds = []
        u_bounds = []
        for _, (lower, upper) in enumerate(self._bounds):
            l_bounds.append(lower)
            u_bounds.append(upper)
        unscaled_samples = self.LHS_sampler.random(n=n_warmup**(self.dim))
        self.x_grid = qmc.scale(unscaled_samples, l_bounds, u_bounds)
    
    def update_indexes(self, ac, gp, y_max, gps_barriers):

        gp_barrier_evaluations = ac(self.x_grid, gp=gp, y_max=y_max, gps = gps_barriers)
        col_mask = np.where(np.isnan(gp_barrier_evaluations),False,True)
        gp_barrier_evaluations[np.logical_not(col_mask)] = -np.inf
        self.most_promising = np.argmax(gp_barrier_evaluations)
        col_mask = np.where(gp_barrier_evaluations < -1e8,False,True)
        # col_mask = np.where(np.isnan(gp_barrier_evaluations),False,True)
        self.best_indexes = np.where(col_mask == True)
    
    def random_best_point(self, random_state):
        idx = random_state.choice(self.best_indexes[0])
        return self.x_grid[idx,:]
    
    def most_promising_point(self):
        return self.x_grid[self.most_promising,:]
    
    def init_LHS(self,init_points):

        l_bounds = []
        u_bounds = []
        for _, (lower, upper) in enumerate(self._bounds):
            l_bounds.append(lower)
            u_bounds.append(upper)
        sampler = qmc.LatinHypercube(self.dim, seed = self.seed)
        self.LHS_sampler = sampler
        sample = sampler.random(n=init_points)
        self.init_sample_scaled = (qmc.scale(sample, l_bounds, u_bounds)).tolist()


    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])


    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return dict(zip(self.keys, x))


    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register_barriers(self, target_barriers):
        '''
        Append the constraint evaluations
        '''
        self._barrier_targets = np.vstack([self._barrier_targets, [target_barriers]])
        return target_barriers

    def register(self, params, target, idx=None):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        params: numpy.ndarray
            A single point, with x.shape[1] == self.dim

        target: float
            Target function value

        idx: int or None, optional (default=None)
            The dataset index of the point to be registered, or None if no dataset is being used

        Returns
        -------
        value: float
            The registered target value

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        if self._debug: print("Registering params", params, "with index", idx, "and target value", target)
        value, info = self.extract_value_and_info(target)

        x_df = pd.DataFrame(params.reshape(1, -1), columns=self._keys, index=[idx], dtype=float)
        self._params = pd.concat((self._params, x_df))
        self._target = np.concatenate([self._target, [value]])
        
        
        if info:  # The return value of the target function is a dict
            if self._target_dict_info.empty:
                # Initialize member
                self._target_dict_info = pd.DataFrame(info, index=[idx])
            else:
                # Append new point to member
                info_new = pd.DataFrame(info, index=[idx])
                self._target_dict_info = pd.concat((self._target_dict_info, info_new))

        if self._debug: print("Point registered successfully")
        return value


    def register_optimization_info(self, info_new):
        """Register relevant information into self._optimization_info"""
        self._optimization_info = pd.concat((self._optimization_info, info_new))
        if self._debug: print("Registered optimization information:", info_new, sep="\n")

    def probe_barriers(self, params, idx = None):
        '''
        Evaluates the barrier functions on a single point x and records them as observations
                Parameters
        ----------
        params: dict
            A single point, with len(x) == self.dim

        idx: int or None, optional (default=None)
            The dataset index of the point to be probed, or None if no dataset is being used

        Returns
        -------
        target_value: float np.array
            Barrier functions values.
        '''
        if self._debug: print("Probing_barriers at point: index {}, value {}".format(idx, params))
        x = self._as_array(params)

        params = dict(zip(self._keys, x))
        evaluations = [self._barrier_functions[i](**params) for i in range(len(self._barrier_functions))]
        target_barriers = np.array(evaluations)
        target_barriers_values = self.register_barriers(target_barriers)

        if self._debug: print("Probed barrier_target values:", target_barriers_values)
        return target_barriers_values


    def probe(self, params, idx=None):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Parameters
        ----------
        params: dict
            A single point, with len(x) == self.dim

        idx: int or None, optional (default=None)
            The dataset index of the point to be probed, or None if no dataset is being used

        Returns
        -------
        target_value: float
            Target function value.
        """
        if self._debug: print("Probing point: index {}, value {}".format(idx, params))
        x = self._as_array(params)

        params = dict(zip(self._keys, x))
        target = self.target_func(**params)
        target_value = self.register(x, target, idx)
        if self._debug: print("Probed target value:", target_value)
        return target_value


    def random_sample(self,size=1):
        """
        Creates random points within the bounds of the space.

        Parameters
        ------------

        size: int
            Number of sample to generate

        Returns
        ----------
        idx: int or None
            The dataset index number of the chosen point, or None if no dataset is being used
        data: numpy.ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        if self.dataset is not None:
            # Recover random row from dataset
            idx = self.random_state.choice(self.dataset.index)
            data = self.dataset.loc[idx, self.keys].to_numpy()
            if self._debug: print("Randomly sampled dataset point: index {}, value {}".format(idx, data))
        else:
            if self._bounds is not None:
                idx = None
                data = np.empty((1, self.dim))
                for col, _ in enumerate(self._bounds):
                    data.T[col] = self.init_sample_scaled[-1][col]

                self.init_sample_scaled.pop()
                if self._debug: print("Uniform randomly sampled point: value {}".format(data))

        return idx, self.array_to_params(data.ravel())


    def max(self):
        """Get maximum target value found and corresponding parameters."""
        if self.barrier_func is None:
            try:
                res = {
                    'target': self.target.max(),
                    'params': dict(
                        zip(self.keys, self.params.values[self.target.argmax()])
                    )
                }
            except ValueError:
                res = {}
        else:
            correct_idxs = np.array(self.in_constraint)
            target_checked = self.target[correct_idxs]
            params_checked = self.params.values[correct_idxs]
            try:
                res = {
                    'target': target_checked.max(),
                    'params': dict(
                        zip(self.keys, params_checked[target_checked.argmax()])
                    )
                }
            except ValueError:
                res = {}            
        return res


    def res(self):
        """Get all target values found and corresponding parameters."""
        params = [dict(zip(self.keys, p)) for p in self.params.values]

        return [
            {"target": target} | param
            for target, param in zip(self.target, params)
        ]


    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds: dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]


    def extract_value_and_info(self, target):
        """
        Return function numeric value and further information

        The return value of the target function can also be a dictionary. In this case,
        we return separately its 'value' field as the true target value, and we also
        return the whole dictionary separately. Otherwise, if the target function is
        purely numeric, we return an empty information dictionary

        Parameters
        ----------
        target: numeric value or dict
            An object returned by the target function

        Returns
        -------
        target: float
            The actual numeric value
        info: dict
            The full input dictionary, or an empty dictionary if target was purely numeric
        """
        if isinstance(target, Number):
            if self._debug: print("Extracting info: target", target, "is scalar")
            return target, {}
        elif isinstance(target, dict):
            key = 'value'
            if key not in target:
                raise ValueError("If target function is a dictionary, it must contain the '{}' field".format(key))
            if self._debug: print("Extracting info: target is a dict with value", target[key])
            return target[key], target
        else:
            raise ValueError("Unrecognized return type '{}' in target function".format(type(target)))


    def initialize_dataset(self, dataset=None, target_column=None):
        """
        Checks and loads the dataset as well as other utilities. The dataset loaded in this class by
        this method is constant and will not change throughout the optimization procedure.

        Parameters
        ----------
        dataset: str, file handle, or pandas.DataFrame, optional (default=None)
            The dataset which constitutes the optimization domain, if any.

        target_column: str, optional (default=None)
            Name of the column that will act as the target value of the optimization.
            Only works if dataset is passed.
        """
        if dataset is None:
            self._dataset = None
            if self._debug: print("initialize_dataset(): dataset is None")
            return

        if isinstance(dataset, pd.DataFrame):
            self._dataset = dataset
        else:
            try:
                self._dataset = pd.read_csv(dataset)
            except:
                raise ValueError("Dataset must be a pandas.DataFrame or a (path to a) valid file")

        if self._debug: print("Shape of initialized dataset is", self._dataset.shape)

        # Check for banned column names
        banned_columns = ('index', 'params', 'target', 'value', 'acquisition', 'ml_mape')
        for col in banned_columns:
            if col in self._dataset.columns:
                raise ValueError("Column name '{}' is not allowed in a dataset, please change it".format(col))

        # Check for relevant class members
        for attr in ('_bounds', '_keys'):
            if not hasattr(self, attr):
                raise ValueError("'self.{}' must be set before initialize_dataset() is called".format(attr))

        # Set target column and check for missing columns
        self._target_column = target_column
        missing_cols = set(self._keys) - set(self._dataset.columns)
        if missing_cols:
            raise ValueError("Columns {} indicated in pbounds are missing "
                             "from the dataset".format(missing_cols))
        if target_column is not None and target_column not in self._dataset:
            raise ValueError("The specified target column '{}' is not present in the dataset".format(target_column))

        # Check that bounds are respected by the corresponding dataset columns
        for key, (lb, ub) in zip(self._keys, self._bounds):
            if self.dataset[key].min() < lb or self.dataset[key].max() >= ub:
                raise ValueError("Dataset values for '{}' column are not consistent with bounds".format(key))


    def find_point_in_dataset(self, params):
        """
        Find index of a matching row in the dataset.

        Parameters
        ----------
        params: dict
            The point to be found in the dataset

        Returns
        -------
        idx: int
            The dataset index of the point found
        target_val: float
            Dataset target value associated to the point found
        """
        dataset_vals = self._dataset[self._keys].values
        x = self.params_to_array(params)

        # Find matching rows and choose randomly one of them
        matches = np.where((dataset_vals == x).all(axis=1))[0]
        if len(matches) == 0:
            raise ValueError("{} not found in dataset".format(params))
        idx = self.random_state.choice(matches)
        target_val = self.dataset.loc[idx, self._target_column]
        if self._debug: print("Located {} as data[{}], with target value {}".format(x, idx, target_val))

        return idx, target_val
