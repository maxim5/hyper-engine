# Hyper-parameters Tuning for Machine Learning

A toolbox for [Model selection](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

Features
--------

* Straight-forward specification

```python
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
```

* Exploration-exploitation trade-off 

Machine learning model selection is expensive.
Each model evaluation requires full training from scratch and may take minutes to hours to days, 
depending on the problem complexity and available computational resources.
*Hyper-Engine* provides the algorithm to explore the space of parameters efficiently, focus on the most promising areas,
thus converge to the maximum as fast as possible.

**Example 1**: the true function is 1-dimensional - `f(x) = x * sin(x)` (black curve) on [-10, 10] interval.
Red dots represent each trial, red curve is the [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) mean,
blue curve is the mean plus or minus one standard deviation.
The optimizer randomly chose the negative mode as more promising.
![1D Bayesian Optimization](https://github.com/maxim5/hyper-engine/raw/master/.images/figure_1.png "Bayesian Optimization")

**Example 2**: the 2-dimensional function `f(x, y) = (x + y) / ((x - 1) ** 2 - sin(y) + 2)` (black curve) on [0, 9]<sup>2</sup> square.
Red dots represent each trial, the Gaussian Process mean and standard deviations are not shown for simplicity.
Note that to achieve the maximum both variables must be picked accurately.
![2D Bayesian Optimization](https://github.com/maxim5/hyper-engine/raw/master/.images/figure_2-1.png "Bayesian Optimization")
![2D Bayesian Optimization](https://github.com/maxim5/hyper-engine/raw/master/.images/figure_2-2.png "Bayesian Optimization")

The code for these and others examples is [here](https://github.com/maxim5/hyper-engine/blob/master/bayesian/strategy_test.py).

Bayesian Optimization
---------------------

Implements the following [methods](https://en.wikipedia.org/wiki/Bayesian_optimization):
- Probability of improvement (See H. J. Kushner. A new method of locating the maximum of an arbitrary multipeak curve in the presence of noise. J. Basic Engineering, 86:97–106, 1964.)
- Expected Improvement (See J. Mockus, V. Tiesis, and A. Zilinskas. Toward Global Optimization, volume 2, chapter The Application of Bayesian Methods for Seeking the Extremum, pages 117–128. Elsevier, 1978)
- [Upper Confidence Bound](http://www.jmlr.org/papers/volume3/auer02a/auer02a.pdf)
- [Mixed / Portfolio strategy](http://mlg.eng.cam.ac.uk/hoffmanm/papers/hoffman:2011.pdf)

Uses [RBF kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) by default, but can be extended.

Finally, can use naive random search.

Installation
------------

*Hyper-Engine* is designed to be ML-platform agnostic, but currently provides only simple [TensorFlow](https://github.com/tensorflow/tensorflow) binding.

Dependencies:
- NumPy
- SciPy
- TensorFlow (optional)
- PyPlot (optional)

Compatibility:
- Python 2.7 (3.5 is coming)
