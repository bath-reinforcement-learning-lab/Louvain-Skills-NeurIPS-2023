import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="louvainskills",
    version="1.0.0",
    author="Joshua Evans",
    author_email="jbe25@bath.ac.uk",
    description="Code accompanying the paper 'Creating Multi-Level Skill Hierarchies in Reinforcement Learning'. Published as a conference paper at NeurIPS 2023.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bath-reinforcement-learning-lab/Louvain-Skills-NeurIPS-2023",
    packages=setuptools.find_packages(exclude=("test")),
    install_requires=[
        "setuptools",
        "autoconf",
        "wheel",
        "numpy",
        "scipy",
        "networkx",
        "igraph",
        "python-igraph",
        "leidenalg",
        "shapely",
        "matplotlib",
        "simpleoptions",
        "simpleenvs",
        "tqdm",
        # "igraph==0.9.11",
        # "python-igraph==0.9.11",
        # "leidenalg==0.8.9",
    ],
    data_files=[
        (
            "gridworld_files",
            [
                "louvainskills/envs/data/xu_four_rooms_brtl.txt",
                "louvainskills/envs/data/xu_four_rooms_trbl.txt",
                "louvainskills/envs/data/nine_rooms_bltr.txt",
                "louvainskills/envs/data/nine_rooms_brtl.txt",
                "louvainskills/envs/data/ramesh_maze_bltr.txt",
            ],
        )
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
)
