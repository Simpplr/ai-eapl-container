import setuptools
from setuptools import setup

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='eapl_nlp',
      version='0.0.1',
      description='Common Analytics components for usage across developed modules',
      author='C V Goudar',
      author_email='cv.goudar@emplay.net',
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False,
      url='https://gitlab.com/emplaysalesgps/eapl_nlp',
      packages=setuptools.find_packages(),
      package_data={},
      scripts=[],
      install_requires=[
          'ftfy==5.8',
          'boto3==1.17.70',
          'torch==1.8.0',
          'spacy==2.3.0',
          'tensorflow==2.7.2',
          'pytextrank==2.0.2',
          'tensorflow_hub==0.8.0',
          'wget==3.2',
          'srt==3.4.1',
          'farm-haystack==0.9.0',
          'fasttext==0.9.2',
          'google==3.0.0',
          'nlp-rake==0.0.2',
          'pyate==0.4.2',
          'sentence-transformers==0.4.1.2',
          'implicit==0.5.2',
          'wordninja==2.0.0',
          'pyspellchecker==0.5.4',
          'simpletransformers==0.60.4',
          'markdownify==0.6.5',
          'pysolr==3.9.0',
          'pytrec_eval==0.5',
          'keybert==0.3.0',
          'profanityfilter==2.0.6',
          'Levenshtein==0.16.0'
      ],
      extras_require={},
      dependency_links=[],
      classifiers=[],
      keywords='',
      )
