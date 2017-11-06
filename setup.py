import os
import setuptools


setuptools.setup(
    name='nowcasting',
    version="0.1.dev0",
    author="Xingjian Shi, Zhihan Gao, Leonard Lausen, Hao Wang, Dit-Yan Yeung, Wang-chun Woo, Wai-kin Wong",
    author_email="xshiab@cse.ust.hk, zhihan.gao@connect.ust.hk, lelausen@connect.ust.hk, hwangaz@cse.ust.hk, dyyeung@cse.ust.hk, wcwoo@hko.gov.hk, wkwong@hko.gov.hk",
    packages=setuptools.find_packages(),
    description='Deep learning for precipitation nowcasting: A benchmark and a new model',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/sxjscience/HKO-7',
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'moviepy', 'numba',
                      'pillow', 'six', 'easydict', 'pyyaml'],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
