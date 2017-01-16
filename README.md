# Hyper-parameters Tuning Engine for Machine Learning problems

Features
--------

* Easy and straight-forward hyper-parameters specification

```
hyper_params_spec = {
  'init_sigma': 10**spec.uniform(-1.5, -1),

  'optimizer': {
    'learning_rate': 10**spec.uniform(-3.2, -2.8),
    'beta1': 0.9,
    'beta2': 0.999,
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

* Approaches exploration-exploitation problem, trying to converge to the maximum as fast as possible

Machine learning model selection is expensive.
Each model evaluation requires full training from scratch and may take minutes to hours to days, 
depending on the problem complexity and available computational resources.
*Hyper-Engine* provides the algorithm to explore the space of parameters efficiently and focus on the most promising areas.

<center>
![1D Bayesian Optimization](https://github.com/maxim5/hyper-engine/raw/master/.images/figure_1.png "Bayesian Optimization")

Optimizing the true function $f(x)=x * sin(x)$ (black).
Red dots represent each trial, red curve is the Gaussian Process mean, blue curve is the mean plus or minus one standard deviation.
</center>
