# Thinking Probabilistically - A Bayesian "Inference" Primer

## All about theoretical:

### Summary

```text
"Statistical" modeling
"Probabilities" and "uncertainty"
Bayes' theorem and statistical inference
"Single" parameter inference and the classic coin-flip problem
Choosing "priors" and why people often "don't like" them, but "should"
Communicating a Bayesian analysis
```

### Expanded Ideas

1. **Statistical Modeling**:
    - Statistical modeling involves creating mathematical representations of real-world processes to predict or explain data. These models help in understanding relationships between variables and in making informed decisions based on data.
    - In practice, statistical models can range from simple linear regression models to complex hierarchical models. They are used in various fields, including economics, biology, engineering, and social sciences. The primary goal is to capture the underlying structure of the data to make predictions or understand the mechanisms driving the observed patterns.

2. **Probabilities and Uncertainty**:
    - Probabilities measure the likelihood of events occurring and are fundamental to handling uncertainty in statistical analysis. Uncertainty reflects the inherent variability and unpredictability in data and events, which probabilistic models aim to quantify and manage.
    - Probabilistic thinking is essential for making decisions under uncertainty. For example, weather forecasts, medical diagnoses, and financial risk assessments all rely on probabilistic models. Understanding the distinction between frequentist and Bayesian interpretations of probability is also key; the former views probability as long-run frequency, while the latter treats it as a degree of belief.

3. **Bayes' Theorem and Statistical Inference**:
    - Bayes' theorem provides a way to update the probability of a hypothesis based on new evidence. Statistical inference using Bayes' theorem involves drawing conclusions about population parameters by incorporating prior beliefs and observed data.
    - Bayes' theorem mathematically expresses how to update the probability of a hypothesis \( P(H|E) \) based on new evidence \( E \): \( P(H|E) = \frac{P(E|H)P(H)}{P(E)} \). Bayesian inference combines this theorem with observed data to refine the estimates of parameters, leading to more accurate and credible conclusions.

4. **Single Parameter Inference and the Classic Coin-Flip Problem**:
    - Single parameter inference focuses on estimating one parameter of interest. The classic coin-flip problem exemplifies this, where the goal is to infer the probability of heads (a single parameter) based on the outcomes of several coin flips.
    - The coin-flip problem is a fundamental example used to illustrate the principles of inference. By observing the outcomes of a series of coin flips, one can infer the probability \( p \) of landing heads using methods such as maximum likelihood estimation or Bayesian inference, where prior beliefs about \( p \) are updated with the observed data.

5. **Choosing Priors and Common Aversion to Them**:
    - Priors represent initial beliefs about parameters before observing data. Selecting appropriate priors is crucial as they influence the results of Bayesian analysis. Despite skepticism and discomfort with subjective priors, they provide a structured way to incorporate prior knowledge into the analysis.
    - The choice of priors can be subjective and controversial, as it introduces personal beliefs into the analysis. Critics argue that this subjectivity can bias results, but proponents highlight that all analyses involve assumptions and that priors can be chosen objectively using empirical data or expert knowledge. Priors can be informative (incorporating specific knowledge) or non-informative (reflecting vague or neutral beliefs).

6. **Communicating a Bayesian Analysis**:
    - Effectively communicating Bayesian analysis involves clearly explaining the choice of priors, the process of updating beliefs with new data, and the interpretation of the results. Transparency and clarity in this communication are essential to ensure that the conclusions are understood and trusted by others.
    - Clear communication in Bayesian analysis involves demystifying the process of selecting priors, showing how data updates these priors, and presenting results in an interpretable way. Visual aids like probability density plots, credible intervals, and posterior distributions are often used to convey the findings effectively. Ensuring that the audience understands the implications of the analysis helps build confidence in the results and supports informed decision-making.

## "Statistics" as a form of modeling

### Summary

--------------------------------------------------------------------------
```text
collecting, organizing, analyzing, and interpreting data
make smt messy to clean and tidy
```

### Expanded Ideas

1. **Statistics as a Form of Modeling**:
    - Statistics is a discipline that uses mathematical and computational techniques to create models representing real-world phenomena. These models help to understand, explain, and predict patterns within data.
    - Statistical modeling involves creating simplified representations of complex systems or processes using mathematical frameworks. These models can be descriptive (summarizing data), inferential (drawing conclusions about populations from samples), or predictive (forecasting future outcomes). Examples include regression models, time series models, and classification models.

2. **Collecting Data**:
    - Data collection is the initial step in statistical analysis, involving gathering information from various sources. This can be done through surveys, experiments, observations, or retrieving data from existing databases. Proper data collection is crucial for the accuracy and reliability of subsequent analysis.
    - Effective data collection strategies ensure that the data is representative and unbiased. This step includes designing appropriate data collection instruments, defining the scope and sample of the study, and ensuring ethical standards are met in the process. For example, in clinical trials, randomized controlled trials are designed to eliminate bias and ensure reliable results.

3. **Organizing Data**:
    - Once collected, data needs to be organized in a systematic way. This involves categorizing, sorting, and structuring data in formats that are easy to analyze. Techniques such as data cleaning, handling missing values, and ensuring consistency are part of this process.
    - Data organization includes creating databases, spreadsheets, or data warehouses where data is stored in a structured format. Techniques such as data normalization (to reduce redundancy) and data transformation (to convert data into suitable formats) are employed. Organized data facilitates easier access, analysis, and interpretation.

4. **Analyzing Data**:
    - Data analysis involves applying statistical methods to uncover patterns, relationships, and trends within the data. This can range from descriptive statistics, like mean and standard deviation, to more complex inferential statistics and predictive modeling techniques.
    - Advanced statistical analysis may involve hypothesis testing, regression analysis, ANOVA, machine learning algorithms, and other techniques to explore relationships within the data. Visualization tools such as charts, graphs, and heatmaps help in understanding the data better and communicating findings effectively.

5. **Interpreting Data**:
    - Interpreting data is the final step where the results of the analysis are examined to draw meaningful conclusions. This involves contextualizing the findings, assessing their significance, and making informed decisions or predictions based on the data.
    - Interpreting data requires a deep understanding of the context and the statistical methods used. It involves evaluating the validity and reliability of the results, considering potential biases, and understanding the practical implications of the findings. Effective interpretation translates statistical results into actionable insights that can influence policy, business strategy, or scientific understanding.

6. **Making Something Messy to Clean and Tidy**:
    - The process of statistical analysis transforms messy, unstructured data into clean, organized, and understandable information. This involves not only technical steps of cleaning and processing data but also synthesizing and presenting the results in a clear, concise manner. The ultimate goal is to provide actionable insights that can inform decisions and strategies.
    - The transformation of messy data into clean and tidy information involves several stages of data preprocessing. This includes data cleaning (removing errors and inconsistencies), data integration (combining data from different sources), and data reduction (simplifying the data without losing essential information). The final presentation of the data often includes summary reports, dashboards, and detailed analyses that are easy to understand and use for decision-making purposes. 

T.B.D
--------------------------------------------------------------------------
    Exploratory Data Analysis (EDA)
        sources: experiments, computer simulations, surveys, field observations, and so on
        • Descriptive statistics
            measures (or statistics)
                summarize or characterize the data in a "quantitative"
                using the mean, mode, standard deviation, interquartile ranges
        • Data visualization
            visually inspecting: histograms, scatter plots ...

--------------------------------------------------------------------------
    Inferential statistics
        data "generalization"
            understand the underlying "mechanism"
                how "generated" the data
                make "predictions"
        rely on probabilistic models
        Models:
            simplified "descriptions" of a given system (or process)
            capture only the most relevant aspects
        Bayesian modeling process:
            1. Given some data | "assumptions" | how it could have been generated
                "approximations" most of the time
            2. "conditioning" the model on our data
            3. check model "makes sense" in different criteria

--------------------------------------------------------------------------
    Probabilities and uncertainty
        - measure that quantifies the "uncertainty"
            + uncertainty is maximum if absence of information
            + world is "deterministic" or "stochastic"
                -> make probabilistic statements
                -> quantify uncertainty
        - binary "outcome"
        - Probabilities are numbers in the interval [0, 1]
        - product rule: p(A, B) = p(A| B) p(B)
        - "conditional" probability vs "unconditioned" probability
            + A and B are independent (unconditioned)
            + knowing B gives us useful information about A (conditional)
        - understanding Bayes' theorem need Conditional probabilities
            p( A| B) = p( A, B) / p(B)

--------------------------------------------------------------------------
    Probability "distributions"
        - mathematical object: describes how "likely" different events are (khả năng xảy ra)
            + restricted somehow to a set of "possible" events
            + think "data generated" from some probability distribution (PD) with "unobserved parameters"
            + Bayes' theorem to "invert" the relationship from data to parameters
        - building blocks of Bayesian models
            "combining" in proper ways to get useful "complex models"
        - famous PD: Gaussian | normal distribution
            + formula: mean | median and mode and standard deviation
            + a random variable:
                can't take any possible value
                values are strictly controlled by PD
                can't predict value but the "probability of value observing"
                COMMON Notation : "distributed as"
            + types of random variable:
                continuous: variables can take "any value" from some interval (float)
                discrete: variables take only "certain values" (int)
            + independent or dependency
                independent: iid variables | identically distributed
                dependency : non iid variables | temporal series
            + trends | seasonal |

--------------------------------------------------------------------------
    Bayes' theorem and statistical inference
            p(H|D) = p(D|H) * p(H) / p(D)
        + H: "hypothesis" or models
        + D: data
            probability of a hypothesis H given the data D
                • p(H): Prior
                    what we know about the value of some
                        parameter before seeing the data D.
                • p(D|H): Likelihood
                    how data introduce in our analysis
                • p(H|D): Posterior -> need calculate
                    the result of the Bayesian analysis
                    a probability distribution
                    balance of the prior and the likelihood
                    vague beliefs (niềm tin mơ hồ)
                    can be updated prior of a new analysis
                • p(D): Evidence
                    marginal likelihood
                    data observing probability averaged over all the possible values
                    care more about relative values vs absolute

--------------------------------------------------------------------------
    Single parameter inference : PAGE_13
        Probabilities are used to measure the uncertainty we have about parameters

        Bayes' theorem:
            + mechanism to correctly update probabilities when new data coming
            + hopefully reducing our uncertainty

        The coin-flipping problem
            • The general model
            • Choosing the likelihood
            • Choosing the prior
            • Getting the posterior
            • Computing and plotting the posterior

            Influence of the prior and how to choose one

        Communicating a Bayesian analysis
            Model notation and visualization
                communicate the model
                Kruschke's diagrams
                
            Summarizing the posterior
                Highest posterior density

        Posterior predictive checks

