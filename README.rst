============================================
Hyper-parameters Tuning for Machine Learning
============================================

- `Overview <#overview>`__
    - `About <#about>`__
    - `Installation <#installation>`__
- `Features <#features>`__
    - `Straight-forward specification <#straight-forward-specification>`__
    - `Exploration-exploitation trade-off <#exploration-exploitation-trade-off>`__
    - `Learning Curve Estimation <#learning-curve-estimation>`__
- `Bayesian Optimization <#bayesian-optimization>`__

--------
Overview
--------

About
=====

*Hyper-Engine* is a toolbox for `model selection and hyper-parameters tuning <https://en.wikipedia.org/wiki/Hyperparameter_optimization>`__.
It aims to provide most state-of-the-art techniques via intuitive API and with minimum dependencies.

Installation
============

.. code-block:: shell

    pip install git+https://github.com/maxim5/hyper-engine.git@master 

Dependencies:

-  NumPy
-  SciPy
-  TensorFlow (optional)
-  PyPlot (optional, only if you're using development version)

Compatibility:

-  Python 2.7 (3.5 is coming)

*Hyper-Engine* is designed to be ML-platform agnostic, but currently provides only simple `TensorFlow <https://github.com/tensorflow/tensorflow>`__ binding.

--------
Features
--------

Straight-forward specification
==============================

.. code-block:: python

    hyper_params_spec = {
      'init_sigma': 10**spec.uniform(-1.5, -1),
      'optimizer': {
        'learning_rate': 10**spec.uniform(-3.5, -2.5),
        'epsilon': 1e-8,
      },
      'conv': {
        'filters': [[3, 3, spec.choice(range(32, 48))],
                    [3, 3, spec.choice(range(72, 96))],
                    [3, 3, spec.choice(range(160, 192))]],
        'activation': spec.choice(['relu', 'leaky_relu', 'prelu', 'elu']),
        'down_sample': {'size': [2, 2], 'pooling': spec.choice(['max_pool', 'avg_pool'])},
        'residual': spec.random_bit(),
        'dropout': spec.uniform(0.75, 1.0),
      },
    }

Exploration-exploitation trade-off
==================================

Machine learning model selection is expensive.
Each model evaluation requires full training from scratch and may take minutes to hours to days, 
depending on the problem complexity and available computational resources.
*Hyper-Engine* provides the algorithm to explore the space of parameters efficiently, focus on the most promising areas,
thus converge to the maximum as fast as possible.

**Example 1**: the true function is 1-dimensional, ``f(x) = x * sin(x)`` (black curve) on [-10, 10] interval.
Red dots represent each trial, red curve is the `Gaussian Process <https://en.wikipedia.org/wiki/Gaussian_process>`__ mean,
blue curve is the mean plus or minus one standard deviation.
The optimizer randomly chose the negative mode as more promising.

.. image:: /.images/figure_1.png
    :width: 80%
    :alt: 1D Bayesian Optimization
    :align: center

**Example 2**: the 2-dimensional function ``f(x, y) = (x + y) / ((x - 1) ** 2 - sin(y) + 2)`` (black surface) on [0,9]x[0,9] square.
Red dots represent each trial, the Gaussian Process mean and standard deviations are not shown for simplicity.
Note that to achieve the maximum both variables must be picked accurately.

.. image:: /.images/figure_2-1.png
   :width: 100%
   :alt: 2D Bayesian Optimization
   :align: center

.. image:: /.images/figure_2-2.png
   :width: 100%
   :alt: 2D Bayesian Optimization
   :align: center

The code for these and others examples is `here <https://github.com/maxim5/hyper-engine/blob/master/hyperengine/tests/strategy_test.py>`__.

Learning Curve Estimation
=========================

*Hyper-Engine* can monitor the model performance during the training and stop early if it's learning too slowly.
This is done via *learning curve prediction*. Note that this technique is compatible with Bayesian Optimization, since
it estimates the model accuracy after full training - this value can be safely used to update Gaussian Process parameters.

Example code:

.. code-block:: python

    curve_params = {
      'burn_in': 30,                # burn-in period: 30 models 
      'min_input_size': 5,          # start predicting after 5 epochs
      'value_limit': 0.80,          # stop if the estimate is less than 80% with high probability
    }
    curve_predictor = LinearCurvePredictor(**curve_params)

Currently there is only one implementation of the predictor, ``LinearCurvePredictor``, 
which is very efficient, but requires relatively large burn-in period to predict model accuracy without flaws.

Note that learning curves can be reused between different models and works quite well for the burn-in,
so it's recommended to serialize and load curve data via ``io_save_dir`` and ``io_load_dir`` parameters.

See also the following paper:
`Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks
by Extrapolation of Learning Curves <http://aad.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf>`__

---------------------
Bayesian Optimization
---------------------

Implements the following `methods <https://en.wikipedia.org/wiki/Bayesian_optimization>`__:

-  Probability of improvement (See H. J. Kushner. A new method of locating the maximum of an arbitrary multipeak curve in the presence of noise. J. Basic Engineering, 86:97–106, 1964.)
-  Expected Improvement (See J. Mockus, V. Tiesis, and A. Zilinskas. Toward Global Optimization, volume 2, chapter The Application of Bayesian Methods for Seeking the Extremum, pages 117–128. Elsevier, 1978)
-  `Upper Confidence Bound <http://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf>`__
-  `Mixed / Portfolio strategy <http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2011.pdf>`__
-  Naive random search.

PI method prefers exploitation to exploration, UCB is the opposite. One of the best strategies we've seen is a mixed one:
start with high probability of UCB and gradually decrease it, increasing PI probability.

Default kernel function used is `RBF kernel <https://en.wikipedia.org/wiki/Radial_basis_function_kernel>`__, but it is extensible.
