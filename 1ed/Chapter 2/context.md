Programming Probabilistically

--------------------------------------------------------------------------
    "Probabilistic programming"
        measure data
        interest parameters
        types of "unknown quantities"
            everything that is "unknown" is treated the same
            assign a probability distribution to it
        Bayesian statistics is a form of "learning"

        computational era | numerical methods
        
        PyMC3
            probabilistic programming languages (PPL) | PPL framework
            probabilistic model | automatically inference
            build complex probabilistic models in a less time-consuming and less error
            focus on model specification and analysis of results nowaday

--------------------------------------------------------------------------
    Inference engines
        Non-Markovian
            faster than Markovian | crude approximation

            Grid computing (brute-force approach)
                Define a reasonable "interval" for the parameter
                Place a grid of points
                For each point in the grid we multiply the likelihood and the prior

                larger number of points will result in a better posterior approximation
                not scale well for many parameters (dimensions)

            Quadratic method (Laplace method or normal approximation)
                approximating the posterior with a Gaussian distribution
                    find the mode of the posterior distribution (optimization methods)
                        find the maximum or minimum of a function
                    estimate the curvature of the function near the mode
                        then compute std of approximating Gaussian

            Variational methods (naive approach)
                run n chains in parallel and then combine the results
                Finding effective ways of parallelizing
                better choice for 
                    large datasets (big data)
                    likelihoods that are too expensive to compute
                quick approximations to the posterior | starting points for MCMC methods
            
                ADVI:
                    Transform : taking the logarithm of a parameter
                    Approximate the unbounded parameters with a Gaussian distribution
                        https://www.sciencedirect.com/topics/mathematics/laplace-approximation
                    Use an optimization method to make 
                        the Gaussian approximation as close as possible to the posterior
                        Evidence Lower Bound (ELBO)

                    already implemented on PyMC3

        Markovian (MCMC family)
            Monte Carlo
            Markov chain
            Metropolis-Hastings
            Hamiltonian Monte Carlo/NUTS
                Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC).
            Other
                Metropolis Coupled MCMC

--------------------------------------------------------------------------
    PyMC3
        library for probabilistic programming
        describe probabilistic models
        using NumPy and Theano

        coin-flipping problem
            Model specification
            inference
            Diagnosing the sampling
            Convergence
            Autocorrelation
            Effective size
        Summarizing the "posterior"
            Posterior-based decisions
            ROPE (Region Of Practical Equivalence)
            Loss functions