# Geomviz: Visualizing Differential Geometry

Goal: Get intuition on Differential Geometry by visualizing manifolds, and learn collaborative coding with GitHub.

- Deadline: Monday 01/30/2023
- Team of 1-4 students.

You will continue the work done by last year's ECE 594n to integrate their contributions into the open-source Python package Geomstats.

## Register your team

First come, first serve. Send a message on slack to register your team.

- [X] Stiefel manifold: Nick Godzik, Jose Nunez, Yequan Zhao and Situo Song.
- [X] Grassmanian manifold: Joshua Kim and Christine Wan
- [X] Manifold of symmetric positive definite (SPD) matrices: Shaun Chen and Qing Yao.
- [X] Special Euclidean group: Karthik Suryanarayana, Zeyu Deng, Monsij Biswal, and Mahmoud Namazi
- [X] Discrete curves: Parsa Madinei & Alireza Parsay.
- [X] Manifold of beta distributions: Ryan Stofer Allen Wang and Marianne Arriola.
- [X] Manifold of categorical distributions: Benedict Lee and Ricky Lee

"Expert" projects: GitHub and Python experience recommended.

- [X] [Klein bottle](https://github.com/geomstats/geomstats/pull/1707/files): Rami Dabit and Terry Wang.
- [X] [Special linear group](https://github.com/geomstats/geomstats/pull/1365/files): Pieter Derksen, Rimika Jaiswal, and Molly Kaplan.
- [X] [Correlation matrices](https://github.com/geomstats/geomstats/pull/1695): Jake Bentley and James McNeice
- [X] [Architecture of the visualization module](https://github.com/geomstats/geomstats/pulls?q=is%3Apr+is%3Aopen+visualization): Amil Khan, Lauren Washington, Sam Feinstein.
- [X] [Polynomial regression](https://github.com/geomstats/geomstats/pull/1605/files): Sean Anderson and Murat Kaan Erdal
- [X] [Discrete surfaces](https://github.com/geomstats/geomstats/pull/1711): Adele Myers.
- [ ] Creating lecture notes for the class: TBD.
- [ ] Continuing a stalled PR from Geomstats: TBD.

## Guidelines

- [Create a GitHub account](https://github.com/).
- [Download and install Anaconda](https://docs.anaconda.com/anaconda/install/index.html).
- Join the slack channel hw_geomviz: this will be where you can easily ask questions to Nina and to the class.

1. Set-up

From a terminal in your computer:

- Clone the GitHub repository **of the class ece594n** with [git clone](https://github.com/git-guides/git-clone).
- In the cloned folder `ece594n`, create a conda environment `conda env create -f environment.yml`.
- Activate the conda environemnt with `conda activate ece594n`.
- Create a new GitHub branch  with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
- Run the code in the folder associated to the manifold you chose (the "manifold folder").
  - The code might not run anymore due to package/module not found: 
    - in this case, use: `import sys; sys.path.append("path/to/folder/with/file")`
  - The code might not run anymore due to (ii) geomstats changes since last year: 
    - in this case, check the names of Python files, classes and functions.

2. Create tests that verify that the code runs correctly.

From the GitHub branch on your computer:

- In your manifold folder, create a file name `test_visualization_[name-of-your-manifold].py`
- In that .py file, copy-paste the content of this [file](https://github.com/geomstats/geomstats/blob/master/tests/tests_geomstats/test_visualization.py).
- Adapt the code so that it tests the **visualization functions of your manifold**: you should have one test function per function in your manifold.
- Test that your tests run using `pytest test_visualization_[name-of-your-manifold].py`.
- Remove portions of code that are not visualization-related and are duplicated from existing code in the geomstats repository.
- Verify that the code follows the Code Structure given in the next section.
- Put your code to international coding style standards using `black [name-of-your-manifold].py` and `isort [name-of-your-manifold].py` and `flake8 [name-of-your-manifold].py`, and similarly for the test file `test_visualization_[name-of-your-manifold].py`.
- Document your code with docstrings, see [here for docstring guidelines](https://github.com/geomstats/geomstats/blob/master/docs/contributing.rst#writing-docstrings).

3. Prepare the notebook for submission to Geomstats
- In your manifold folder, edit the .ipynb to change everything that you do not understand.
- If there exists a notebook on geomstats on a similar topic, create one notebook that merges the two.

4. Submit your Pull Request.
- Clone the GitHub repository **[of Geomstats](https://github.com/geomstats/geomstats)** with [git clone](https://github.com/git-guides/git-clone).
- Create a new GitHub branch **[of Geomstats]** on your computer with [git checkout -b](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches).
- **Expert project only:** you might need to pull an existing branch into your new GitHub branch:
  - Add the remote repository (different from origin): `git remote add ninamiolane git@github.com:ninamiolane/geomstats.git`
  - From your local branch: `git pull ninamiolane the-existing-branch`.
- On your branch, in the Geomstats' folder [visualization](https://github.com/geomstats/geomstats/tree/master/geomstats/visualization): add your `[name-of-your-manifold].py`.
- On your branch, in the Geomstats' folder [tests_geomstats](https://github.com/geomstats/geomstats/tree/master/tests/tests_geomstats): add your `test_visualization_[name-of-your-manifold].py`
- Submit your work as a [Pull Request (PR)](https://opensource.com/article/19/7/create-pull-request-github) to the Geomstats GitHub repository.
- You can submit code to your PR (i.e., modify your PR) anytime until the deadline.
- Your code should pass the GitHub Action tests: it will automatically verify that your code runs.
- Your code will receive a code review. Address the review, and resubmit.

![Using GitHub.](/lectures/figs/github.png)

## Information about Code Structure 

Design from Elodie Maignant, PhD student at INRIA (France) and contributor to Geomstats.

− In your manifold folder, you should see a file `[name-of-your-manifold].py`:
  - With a python class named after your manifold `class YourManifold`.
  - With visualization utils as methods from this class, such as:
    - `plot`: draws the manifold (e.g. cone for SPD matrices)
    - `plot_grid`: draws the manifold with a geodesic grid.
    - `plot_rendering`: draws the manifold with regularly sampled data.
    - `plot_tangent_space`: draws the tangent space to the manifold at a point.
    - `scatter`: plot a point cloud. Inherits matplotlib.scatter parameters.
    
    - `plot_geodesic` allows to visualise a (discretised) geodesic. Takes either point and tangent vec as parameters, or initial point and end point as parameters.
    
    − `plot_vector_field` allows to visualise vectors fields. Takes points and tangent vecs as parameters.


## Grading Criteria

- The code style is clean: it passes the GitHub Actions' **Linting** (10%).
- The code is properly documented following the documentation guidelines (10%).
- The code runs: it passes the GitHub Actions' **Testing** (20%).
- There is one test function per visualization function (20%).
- You have made substantial improvements compared to the current version of the code, or you chose an expert project (20%).
- You have addressed the comments in the code review and merged into Geomstats (20%).
