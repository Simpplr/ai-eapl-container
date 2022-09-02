import setuptools
from setuptools import setup

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='eapl',
      version='0.0.1',
      description='Common Analytics components for usage across developed modules',
      # packages=['eapl'],
      author='C V Goudar',
      author_email='cv.goudar@emplay.net',
      long_description=long_description,
      long_description_content_type='text/markdown'
      , zip_safe=False,
      url='https://gitlab.com/emplaysalesgps/eapl',
      packages=setuptools.find_packages(),
      package_data={},
      scripts=[],
      install_requires=[
          'pandas==1.1.0',
          'numpy>=1.18.5',
          'fuzzywuzzy==0.18.0',
          'scikit-learn==0.23.2',
          'treeinterpreter==0.2.2',
          'sklearn_pandas==2.0.0',
          'mlxtend==0.18.0'

      ],
      extras_require={},
      dependency_links=[],
      classifiers=[],
      keywords='',
      )
