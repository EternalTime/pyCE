from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name = 'pyCE',
      version = '0.0.1',
      description = 'A python library for Configurational Entropy projects',
      long_description = readme(),
      url = 'https://github.com/EternalTime/pyCE',
      author = 'Damian R Sowinski',
      author_email = 'damian.sowinski@dartmouth.edu',
      classifiers = [
        'Development Status :: Pre-use',
        'Intended Audience :: Physicists',
        'Topic :: Field Theory :: Cosmology :: Information Theory',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
      ],
      license = 'MIT',
      packages = find_packages(),
      install_requires=[
          'tqdm',
      ],
      include_package_data=True,
      zip_safe=False)
